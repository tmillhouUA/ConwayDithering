// Main thread coordinator — manages the worker pool, outer GA loop,
// canvas rendering, and progress reporting.

const MARGIN    = 10;
const OFFSETS   = [0, 25, 13, 38];    // 0, 1/2, 1/4, 3/4 of STEP (rounded)
// STEP, PAD, TILE_SIZE are runtime variables — set in prepareAndRun from slider
let STEP      = 50;
let PAD       = MARGIN + Math.max(...OFFSETS) + MARGIN;
let TILE_SIZE = STEP + 2 * MARGIN;
// OUTER_ITS is now read live from the control panel
const MAX_WORKERS = Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) * 0.75));

let workers    = [];
let workerBusy = [];
let jobQueue   = [];
let pendingTiles = 0;
let outerItResolver = null;
let pendingPrecursorWrites = [];

let currentRunId = null;
let currentRunFilename = null;
const expandedLogIds = new Set();

let srcW, srcH, paddedW, paddedH;
let ySteps, xSteps;
let precursorGrid;      // Uint8Array — full padded precursor state
let bestPrecursorGrid;  // Uint8Array — precursor at best fitness so far
let paddedTarget;    // Float32Array — full padded target in [0.25, 1.0]
let outputImageData; // ImageData written to canvas
let bestImageData;   // ImageData of best frame so far
let bestFitness = Infinity;
let canvas, ctx, progressEl, uploadArea;
let startPauseBtn, saveCurrentBtn, saveBestBtn;
let plotCanvas, plotCtx;
let fitnessHistory = [];        // avg loss per iteration for active function
let allLossHistory = [[], [], []]; // avg loss per iteration for all three [L2, L1, Huber]
let tileFitnessSum, tileFitnessCount;
let tileAllLossSums, tileAllLossCount;
let isPaused = true;
let pauseResolver = null;
let currentOuterIt = 0;

// History snapshots: saved every 10 iterations
// Each entry: { it: number, precursorGrid: Uint8Array, golSteps: number }
let historySnapshots = [];
let activeTab = 'output'; // 'output' | 'history'

// ------------------------------------------------------------
// Control panel helpers
// ------------------------------------------------------------

function getParams() {
    return {
        outerIts:    parseInt(document.getElementById('imgItsSlider').value),
        pop:         parseInt(document.getElementById('popSlider').value),
        tileIts:     parseInt(document.getElementById('tileItsSlider').value),
        golSteps:    parseInt(document.getElementById('golStepsSlider').value),
        mutRateStart: Math.pow(10, parseFloat(document.getElementById('mutRateSlider').value)),
        elitism:     parseInt(document.getElementById('elitismSlider').value),
        revertCycles:  parseInt(document.getElementById('revertSlider').value) / 4,
        decayLambda:   parseInt(document.getElementById('decayLambdaSlider').value),
        lossFunc:    parseInt(document.getElementById('lossFuncSelect').value),
    };
}

const PARAM_KEY = 'conwayDitherParams';

const PARAM_CONTROLS = [
    { slider: 'popSlider',         val: 'popVal',         fmt: v => v },
    { slider: 'tileSizeSlider',    val: 'tileSizeVal',    fmt: v => v },
    { slider: 'tileItsSlider',     val: 'tileItsVal',     fmt: v => v },
    { slider: 'imgItsSlider',      val: 'imgItsVal',      fmt: v => v },
    { slider: 'golStepsSlider',    val: 'golStepsVal',    fmt: v => v },
    { slider: 'mutRateSlider',     val: 'mutRateVal',     fmt: v => Math.pow(10, parseFloat(v)).toExponential(1) },
    { slider: 'decayLambdaSlider', val: 'decayLambdaVal', fmt: v => v },
    { slider: 'revertSlider',      val: 'revertVal',      fmt: v => v == 0 ? 'off' : `${v} its` },
    { slider: 'elitismSlider',     val: 'elitismVal',     fmt: v => v == 0 ? 'off' : `${v}` },
    { select: 'lossFuncSelect' },
];

function saveParams() {
    const out = {};
    for (const c of PARAM_CONTROLS) {
        if (c.slider) out[c.slider] = document.getElementById(c.slider).value;
        if (c.select) out[c.select] = document.getElementById(c.select).value;
    }
    try { localStorage.setItem(PARAM_KEY, JSON.stringify(out)); } catch {}
}

function loadParams() {
    try { return JSON.parse(localStorage.getItem(PARAM_KEY)) || {}; } catch { return {}; }
}

function applyParams(saved) {
    for (const c of PARAM_CONTROLS) {
        if (c.slider) {
            const el = document.getElementById(c.slider);
            const display = document.getElementById(c.val);
            if (saved[c.slider] !== undefined) {
                el.value = saved[c.slider];
                display.textContent = c.fmt(el.value);
            }
        }
        if (c.select && saved[c.select] !== undefined) {
            document.getElementById(c.select).value = saved[c.select];
        }
    }
}

function resetParams() {
    if (!confirm('Reset all parameters to defaults?')) return;
    localStorage.removeItem(PARAM_KEY);
    for (const c of PARAM_CONTROLS) {
        if (c.slider) {
            const el = document.getElementById(c.slider);
            el.value = el.defaultValue;
            document.getElementById(c.val).textContent = c.fmt(el.value);
        }
        if (c.select) {
            const el = document.getElementById(c.select);
            el.value = el.options[0].value;
        }
    }
    updateElitismMax();
}

function initControlPanel() {
    for (const c of PARAM_CONTROLS) {
        if (c.slider) {
            const el = document.getElementById(c.slider);
            const display = document.getElementById(c.val);
            el.addEventListener('input', () => { display.textContent = c.fmt(el.value); saveParams(); });
        }
        if (c.select) {
            document.getElementById(c.select).addEventListener('change', saveParams);
        }
    }
    document.getElementById('popSlider').addEventListener('input', updateElitismMax);
    updateElitismMax();

    document.getElementById('lossFuncSelect').addEventListener('change', drawPlot);

    applyParams(loadParams());
    updateElitismMax();
    saveParams();
}

function updateElitismMax() {
    const popSlider     = document.getElementById('popSlider');
    const elitismSlider = document.getElementById('elitismSlider');
    const elitismVal    = document.getElementById('elitismVal');
    const maxElitism = parseInt(popSlider.value) - 1;
    elitismSlider.max = maxElitism;
    if (parseInt(elitismSlider.value) > maxElitism) {
        elitismSlider.value = maxElitism;
        elitismVal.textContent = maxElitism || 'off';
    }
}

// ------------------------------------------------------------
// Entry point — called from index.html after DOM ready
// ------------------------------------------------------------

function init() {
    canvas        = document.getElementById('canvas');
    ctx           = canvas.getContext('2d');
    plotCanvas    = document.getElementById('plot');
    plotCtx       = plotCanvas.getContext('2d');
    new ResizeObserver(() => {
        const h = plotCanvas.clientHeight;
        if (h > 0 && plotCanvas.height !== h) {
            plotCanvas.height = h;
            drawPlot();
        }
    }).observe(plotCanvas);
    progressEl    = document.getElementById('progress');
    startPauseBtn = document.getElementById('startPauseBtn');
    saveCurrentBtn = document.getElementById('saveCurrentBtn');
    saveBestBtn   = document.getElementById('saveBestBtn');
    uploadArea    = document.getElementById('uploadArea');

    startPauseBtn.addEventListener('click', onStartPause);
    saveCurrentBtn.addEventListener('click', onSaveCurrent);
    saveBestBtn.addEventListener('click', onSaveBest);

    initLoupe();

    document.getElementById('tabOutput').addEventListener('click', () => switchTab('output'));
    document.getElementById('tabHistory').addEventListener('click', () => switchTab('history'));

    document.getElementById('tabOverview').addEventListener('click', () => switchInfoTab('overview'));
    document.getElementById('tabInstructions').addEventListener('click', () => switchInfoTab('instructions'));
    document.getElementById('tabLogs').addEventListener('click', () => switchInfoTab('logs'));

    document.getElementById('historyItSlider').addEventListener('input', onHistorySliderChange);
    document.getElementById('historyGolSlider').addEventListener('input', onHistorySliderChange);
    document.getElementById('historySaveCurrentBtn').addEventListener('click', onHistorySaveCurrent);
    document.getElementById('historySaveSeriesBtn').addEventListener('click', onHistorySaveSeries);

    // Register service worker for COOP/COEP headers (enables SharedArrayBuffer)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./sw.js?v=1').then(reg => {
            if (!navigator.serviceWorker.controller) {
                // First activation — reload once to get headers applied
                window.location.reload();
            }
        });
    }

    document.getElementById('fileInput').addEventListener('change', onFileSelected);
    document.getElementById('testImageBtn').addEventListener('click', onTestImage);

    initControlPanel();
    spawnWorkers();
    renderLog();
}

// ------------------------------------------------------------
// Worker pool
// ------------------------------------------------------------

function spawnWorkers() {
    for (let i = 0; i < MAX_WORKERS; i++) {
        const w = new Worker('./worker.js');
        w.onmessage = (e) => handleWorkerMessage(i, e.data);
        workers.push(w);
        workerBusy.push(false);
    }
}

function allWorkersReady() {
    window._workerReadyCount = (window._workerReadyCount || 0) + 1;
    if (window._workerReadyCount >= MAX_WORKERS) {
        document.getElementById('fileInput').disabled = false;
        document.getElementById('testImageBtn').disabled = false;
        document.getElementById('statusText').textContent = 'Ready.';
    }
}

// ------------------------------------------------------------
// File loading
// ------------------------------------------------------------

function onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    currentRunFilename = file.name;
    const img = new Image();
    img.onload = () => prepareAndRun(img);
    img.src = URL.createObjectURL(file);
}

function onTestImage() {
    currentRunFilename = 'dennett.png';
    const img = new Image();
    img.onload = () => prepareAndRun(img);
    img.src = './dennett.png';
}

function prepareAndRun(img) {
    // Lock in tile geometry from slider (category 2 — only at start)
    STEP      = parseInt(document.getElementById('tileSizeSlider').value);
    PAD       = MARGIN + Math.max(...OFFSETS) + MARGIN;
    TILE_SIZE = STEP + 2 * MARGIN;

    const offscreen = document.createElement('canvas');
    offscreen.width  = img.naturalWidth;
    offscreen.height = img.naturalHeight;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(img, 0, 0);
    const imageData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);

    srcW = offscreen.width;
    srcH = offscreen.height;
    const maxOffset = Math.max(...OFFSETS);
    ySteps = Math.ceil((srcH + maxOffset) / STEP);
    xSteps = Math.ceil((srcW + maxOffset) / STEP);
    // Pad based on tile coverage, not raw src size, so edge tiles don't read past the buffer
    paddedW = xSteps * STEP + 2 * PAD;
    paddedH = ySteps * STEP + 2 * PAD;

    // Build padded float target [0.25, 1.0], 1.0 = dead = white border
    paddedTarget = new Float32Array(paddedH * paddedW).fill(1.0);
    for (let r = 0; r < srcH; r++) {
        for (let c = 0; c < srcW; c++) {
            const i = (r * srcW + c) * 4;
            const lum = 0.2126 * imageData.data[i] + 0.7152 * imageData.data[i+1] + 0.0722 * imageData.data[i+2];
            paddedTarget[(r + PAD) * paddedW + (c + PAD)] = (lum / 255.0) * 0.75 + 0.25;
        }
    }

    // Initialize precursor grid: random binary, border pinned dead (1)
    precursorGrid = new Uint8Array(paddedH * paddedW);
    for (let i = 0; i < precursorGrid.length; i++)
        precursorGrid[i] = Math.random() < 0.5 ? 0 : 1;
    pinBorder(precursorGrid, paddedW, paddedH, PAD);

    // Set up output canvas
    canvas.width  = srcW;
    canvas.height = srcH;
    outputImageData = ctx.createImageData(srcW, srcH);
    outputImageData.data.fill(255);
    ctx.putImageData(outputImageData, 0, 0);
    sizeCanvas();

    document.getElementById('fileInput').disabled = true;
    uploadArea.style.display = 'none';
    document.getElementById('viewer').style.display = 'flex';
    requestAnimationFrame(() => {
        const outputH = document.getElementById('outputPanel').offsetHeight;
        document.getElementById('infoPanel').style.height = outputH + 'px';
        document.getElementById('leftCol').style.height   = outputH + 'px';
    });
    fitnessHistory = [];
    allLossHistory = [[], [], []];
    bestFitness = Infinity;
    bestPrecursorGrid = null;
    bestImageData = null;
    historySnapshots = [];
    isPaused = true;
    currentOuterIt = 0;
    activeTab = 'output';
    switchTab('output');
    startPauseBtn.textContent = 'Start';
    startPauseBtn.disabled = false;
    saveCurrentBtn.disabled = false;
    saveBestBtn.disabled = true;
    setControlState('pre-start');
    drawPlot();
    drawPrecursorToCanvas();

    currentRunId = crypto.randomUUID();
    const initParams = getParams();
    logUpsert({
        id: currentRunId,
        filename: currentRunFilename || 'unknown',
        startedAt: new Date().toISOString(),
        width: srcW,
        height: srcH,
        params: {
            outerIts:     initParams.outerIts,
            pop:          initParams.pop,
            tileSize:     parseInt(document.getElementById('tileSizeSlider').value),
            tileIts:      initParams.tileIts,
            golSteps:     initParams.golSteps,
            mutRateStart: initParams.mutRateStart,
            elitism:      initParams.elitism,
            revertCycles: initParams.revertCycles,
            decayLambda:  initParams.decayLambda,
            lossFunc:     initParams.lossFunc,
        },
        losses:     { l2: null, l1: null, huber: null },
        bestLosses: { l2: null, l1: null, huber: null },
        iterations: 0,
        status: 'waiting',
    });
    renderLog(true);

    runAllIterations();
}

function drawPrecursorToCanvas() {
    for (let r = 0; r < srcH; r++) {
        for (let c = 0; c < srcW; c++) {
            const v = precursorGrid[(r + PAD) * paddedW + (c + PAD)] === 0 ? 0 : 255;
            const idx = (r * srcW + c) * 4;
            outputImageData.data[idx]     = v;
            outputImageData.data[idx + 1] = v;
            outputImageData.data[idx + 2] = v;
            outputImageData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(outputImageData, 0, 0);
}

function pinBorder(grid, W, H, margin) {
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            if (r < margin || r >= H - margin || c < margin || c >= W - margin)
                grid[r * W + c] = 1;
        }
    }
}

// ------------------------------------------------------------
// Main GA loop
// ------------------------------------------------------------

async function runAllIterations() {
    // Start paused — wait for user to press Start
    await waitIfPaused();

    const p = getParams();
    const outerIts = p.outerIts;

    let consecutiveBadCycles = 0;
    let mutDecayIt = 0;  // only advances on accepted cycles

    for (currentOuterIt = 0; currentOuterIt < outerIts; currentOuterIt++) {
        // At the start of each cycle: revert if bad cycle count has reached the threshold
        const revertCycles = getParams().revertCycles;
        if ((currentOuterIt % 4) === 0 && revertCycles > 0
                && consecutiveBadCycles >= revertCycles && bestPrecursorGrid !== null) {
            precursorGrid.set(bestPrecursorGrid);
            consecutiveBadCycles = 0;
        }

        const params = getParams();
        const MUT_RATE_DECAY = Math.pow(2, -1 / params.decayLambda);
        const mutRate = params.mutRateStart * Math.pow(MUT_RATE_DECAY, mutDecayIt);
        mutDecayIt++;
        await runOneIteration(currentOuterIt, mutRate, params);

        // Always commit tile writes mid-cycle so next offset warm-starts from this result
        for (const w of pendingPrecursorWrites)
            writeTileToPrecursorGrid(w.tileX, w.tileY, w.offset, w.precursorOut);
        pinBorder(precursorGrid, paddedW, paddedH, PAD);

        const avgFitness = fitnessHistory[fitnessHistory.length - 1];

        // Check every full 4-iteration cycle (one pass through all offsets)
        if ((currentOuterIt % 4) === 3) {
            const cycleAvg = fitnessHistory.slice(-4).reduce((a, b) => a + b, 0) / 4;
            if (cycleAvg < bestFitness) {
                bestFitness = cycleAvg;
                bestPrecursorGrid = precursorGrid.slice();
                bestImageData = ctx.getImageData(0, 0, srcW, srcH);
                saveBestBtn.disabled = false;
                consecutiveBadCycles = 0;
            } else {
                consecutiveBadCycles++;
            }
        }

        // Save snapshot when L2 loss drops >10% from last snapshot, or >25 its since last snapshot
        const l2Loss = allLossHistory[0][allLossHistory[0].length - 1];
        const lastSnap = historySnapshots[historySnapshots.length - 1];
        const lastSnapLoss = lastSnap?._loss ?? Infinity;
        const lastSnapIt   = lastSnap?.it ?? -Infinity;
        const itsSinceLast = (currentOuterIt + 1) - lastSnapIt;
        if (l2Loss < lastSnapLoss * 0.9 || itsSinceLast >= 32) {
            const snap = {
                it: currentOuterIt + 1,
                precursorGrid: precursorGrid.slice(),
                golSteps: getParams().golSteps,
                _loss: l2Loss,
            };
            historySnapshots.push(snap);
            updateHistorySliders();
        }

        const pct = Math.round((currentOuterIt + 1) / outerIts * 100);
        progressEl.textContent = `It. ${currentOuterIt + 1} / ${outerIts} — ${pct}% — Log Loss: ${Math.log10(Math.max(avgFitness, 1e-6)).toFixed(2)}`;

        await waitIfPaused();
    }

    startPauseBtn.textContent = 'Done';
    startPauseBtn.disabled = true;
    setControlState('done');
    progressEl.textContent = `Done — Log Loss: ${Math.log10(Math.max(bestFitness, 1e-6)).toFixed(2)}`;
    updateRunLog('complete');
}

// Category definitions — IDs of controls in each group
const CAT2_IDS = ['tileSizeSlider', 'imgItsSlider', 'golStepsSlider'];
const CAT3_IDS = ['popSlider', 'tileItsSlider', 'mutRateSlider', 'decayLambdaSlider', 'revertSlider', 'elitismSlider'];

function setControlState(state) {
    // state: 'pre-start' | 'running' | 'paused' | 'done'
    const cat2Disabled = state !== 'pre-start';
    const cat3Disabled = state === 'running' || state === 'done';
    for (const id of CAT2_IDS)
        document.getElementById(id).disabled = cat2Disabled;
    for (const id of CAT3_IDS)
        document.getElementById(id).disabled = cat3Disabled;
    document.getElementById('resetParamsBtn').disabled = state !== 'pre-start';
}

function waitIfPaused() {
    if (!isPaused) return Promise.resolve();
    return new Promise(resolve => { pauseResolver = resolve; });
}

function onStartPause() {
    if (isPaused) {
        isPaused = false;
        startPauseBtn.textContent = 'Pause';
        setControlState('running');
        if (pauseResolver) { pauseResolver(); pauseResolver = null; }
        updateRunLog('running');
    } else {
        isPaused = true;
        startPauseBtn.textContent = 'Resume';
        setControlState('paused');
        updateRunLog('paused');
    }
}

function runOneIteration(outerIt, mutRate, params) {
    return new Promise(resolve => {
        outerItResolver = resolve;

        // Cycle through 4 tile grid offsets each iteration
        const offset = OFFSETS[outerIt % 4];

        // Start at ty/tx = -1 so tiles at non-zero offsets still cover the top/left edge
        const tyStart = -1, txStart = -1;
        const tyEnd = ySteps, txEnd = xSteps;
        pendingTiles = (tyEnd - tyStart) * (txEnd - txStart);
        pendingPrecursorWrites = [];
        jobQueue = [];
        tileFitnessSum = 0;
        tileFitnessCount = 0;
        tileAllLossSums = [0, 0, 0];
        tileAllLossCount = 0;

        for (let ty = tyStart; ty < tyEnd; ty++) {
            for (let tx = txStart; tx < txEnd; tx++) {
                const targetTile  = extractTileF32(paddedTarget,  tx, ty, offset);
                const precursorIn = extractTileU8 (precursorGrid, tx, ty, offset);
                jobQueue.push({
                    outerIt, tileX: tx, tileY: ty, offset,
                    targetTile, precursorIn,
                    tileW: TILE_SIZE, tileH: TILE_SIZE,
                    mutRate,
                    pop:      params.pop,
                    gens:     params.tileIts,
                    nIts:     params.golSteps,
                    elitism:  params.elitism,
                    lossFunc: params.lossFunc,
                    seed: ((outerIt * 10000 + ty * 1000 + tx) >>> 0) || 1
                });
            }
        }

        for (let i = 0; i < workers.length; i++) dispatchNext(i);
    });
}

function dispatchNext(workerIdx) {
    if (jobQueue.length === 0) return;
    const job = jobQueue.shift();
    workerBusy[workerIdx] = true;
    workers[workerIdx].postMessage(job, [job.targetTile.buffer, job.precursorIn.buffer]);
}

// ------------------------------------------------------------
// Tile extraction helpers
// ------------------------------------------------------------

// Tile origin in padded space: tiles are centered on canvas pixels.
// On even iterations (offset=0): tile (0,0) center = canvas (0,0)
//   => tile starts at padded row (PAD - MARGIN) = HALF_STEP
// On odd iterations (offset=HALF_STEP): tile (0,0) center = canvas (HALF_STEP, HALF_STEP)
//   => tile starts at padded row (PAD - MARGIN + HALF_STEP) = 2*HALF_STEP

function tileOrigin(tx, ty, offset) {
    // Tile starts MARGIN cells before its center in padded space.
    // Center of tile (0,0) with offset=0 maps to canvas (0,0), which is PAD into padded space.
    // So tile origin = PAD - MARGIN + ty*STEP + offset
    const base = PAD - MARGIN;  // = max(OFFSETS) = 38
    const row = base + ty * STEP + offset;
    const col = base + tx * STEP + offset;
    return { row, col };
}

function extractTileF32(grid, tx, ty, offset) {
    const tile = new Float32Array(TILE_SIZE * TILE_SIZE);
    const { row, col } = tileOrigin(tx, ty, offset);
    for (let r = 0; r < TILE_SIZE; r++) {
        const srcOff = (row + r) * paddedW + col;
        tile.set(grid.subarray(srcOff, srcOff + TILE_SIZE), r * TILE_SIZE);
    }
    return tile;
}

function extractTileU8(grid, tx, ty, offset) {
    const tile = new Uint8Array(TILE_SIZE * TILE_SIZE);
    const { row, col } = tileOrigin(tx, ty, offset);
    for (let r = 0; r < TILE_SIZE; r++) {
        const srcOff = (row + r) * paddedW + col;
        tile.set(grid.subarray(srcOff, srcOff + TILE_SIZE), r * TILE_SIZE);
    }
    return tile;
}

// ------------------------------------------------------------
// Worker message handler
// ------------------------------------------------------------

function handleWorkerMessage(workerIdx, msg) {
    if (msg.type === 'ready') {
        allWorkersReady();
        return;
    }

    if (msg.type === 'tile_done') {
        workerBusy[workerIdx] = false;

        writeTileToCanvas(msg.tileX, msg.tileY, msg.offset, msg.evolvedOut);
        pendingPrecursorWrites.push({ tileX: msg.tileX, tileY: msg.tileY, offset: msg.offset, precursorOut: msg.precursorOut });
        tileFitnessSum += msg.fitness;
        tileFitnessCount++;
        for (let k = 0; k < 3; k++) tileAllLossSums[k] += msg.allLosses[k];
        tileAllLossCount++;

        pendingTiles--;
        if (pendingTiles === 0) {
            fitnessHistory.push(tileFitnessSum / tileFitnessCount);
            for (let k = 0; k < 3; k++)
                allLossHistory[k].push(tileAllLossSums[k] / tileAllLossCount);
            ctx.putImageData(outputImageData, 0, 0);
            drawPlot();
            updateRunLog();
            outerItResolver();
        } else {
            dispatchNext(workerIdx);
        }
    }
}

// ------------------------------------------------------------
// Canvas and precursor grid writes
// ------------------------------------------------------------

function writeTileToCanvas(tx, ty, offset, evolvedOut) {
    // Canvas origin for this tile's center = tileOrigin + MARGIN - PAD
    const { row: tRow, col: tCol } = tileOrigin(tx, ty, offset);
    const canvasRow0 = tRow + MARGIN - PAD;
    const canvasCol0 = tCol + MARGIN - PAD;
    for (let r = 0; r < STEP; r++) {
        for (let c = 0; c < STEP; c++) {
            const srcIdx = (r + MARGIN) * TILE_SIZE + (c + MARGIN);
            const dstRow = canvasRow0 + r;
            const dstCol = canvasCol0 + c;
            if (dstRow < 0 || dstRow >= srcH || dstCol < 0 || dstCol >= srcW) continue;
            const dstIdx = (dstRow * srcW + dstCol) * 4;
            const v = evolvedOut[srcIdx] * 255;
            outputImageData.data[dstIdx]     = v;
            outputImageData.data[dstIdx + 1] = v;
            outputImageData.data[dstIdx + 2] = v;
            outputImageData.data[dstIdx + 3] = 255;
        }
    }
}

function writeTileToPrecursorGrid(tx, ty, offset, precursorOut) {
    const { row, col } = tileOrigin(tx, ty, offset);
    for (let r = 0; r < TILE_SIZE; r++) {
        precursorGrid.set(
            precursorOut.subarray(r * TILE_SIZE, (r + 1) * TILE_SIZE),
            (row + r) * paddedW + col
        );
    }
}

// ------------------------------------------------------------
// Canvas display sizing
// ------------------------------------------------------------

function sizeCanvas() {
    // Vertical budget: viewport minus header (~60px), controls (~80px), padding and gaps (~40px), then 10% shorter
    const maxH = Math.max(200, (window.innerHeight - 180) * 0.9);
    // Horizontal budget: viewport minus two 300px side columns, two 1rem gaps, and 2rem body padding
    const maxW = Math.max(200, window.innerWidth - 600 - 4 * 16);
    const scale = Math.min(maxW / srcW, maxH / srcH);
    canvas.style.width  = Math.round(srcW * scale) + 'px';
    canvas.style.height = Math.round(srcH * scale) + 'px';

}

// ------------------------------------------------------------
// Fitness plot
// ------------------------------------------------------------

const LOSS_COLORS = ['#7af', '#fa7', '#7f7'];  // L2=blue, L1=orange, Huber=green
const LOSS_NAMES  = ['Euclidean', 'Manhattan', 'Huber'];

function smoothEMA(history) {
    const alpha = 0.14;
    const out = [];
    for (let i = 0; i < history.length; i++)
        out.push(i === 0 ? history[0] : alpha * history[i] + (1 - alpha) * out[i - 1]);
    return out;
}

function drawPlot() {
    const W = plotCanvas.width;
    const H = plotCanvas.height;
    const PAD_L = 42, PAD_R = 10, PAD_T = 18, PAD_B = 52;
    const pw = W - PAD_L - PAD_R;
    const ph = H - PAD_T - PAD_B;

    plotCtx.fillStyle = '#111';
    plotCtx.fillRect(0, 0, W, H);

    const activeLoss = parseInt(document.getElementById('lossFuncSelect').value);

    // Y range — computed from data if available, otherwise dummy range for empty grid
    let yMin = 0, yMax = 1, xMax = 1;
    let shifted = [[], [], []];
    let activeLog = [];
    const hasData = fitnessHistory.length > 0;

    if (hasData) {
        const smoothed = allLossHistory.map(h => h.length ? smoothEMA(h) : []);
        const logAll   = smoothed.map(s => s.map(v => Math.log10(Math.max(v, 1e-6))));
        activeLog = logAll[activeLoss];
        const offset0 = activeLog.length ? activeLog[0] : 0;
        shifted = logAll.map((vals, k) => {
            if (!vals.length) return [];
            const delta = k === activeLoss ? 0 : offset0 - vals[0];
            return vals.map(v => v + delta);
        });
        const allVals = shifted.flatMap(s => s);
        const dataMin = Math.min(...allVals);
        const dataMax = Math.max(...allVals);
        const dataRange = Math.max(dataMax - dataMin, 0.1);
        const yMargin = dataRange * 0.125;
        yMin = dataMin - yMargin;
        yMax = dataMax + yMargin;
        xMax = fitnessHistory.length / 0.9;
    }

    // Grid lines + Y axis tick numbers (only when data exists)
    plotCtx.strokeStyle = '#333';
    plotCtx.lineWidth = 1;
    plotCtx.font = '10px monospace';
    plotCtx.textAlign = 'right';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
        const y = PAD_T + ph - (i / yTicks) * ph;
        plotCtx.beginPath();
        plotCtx.moveTo(PAD_L, y);
        plotCtx.lineTo(PAD_L + pw, y);
        plotCtx.strokeStyle = '#333';
        plotCtx.stroke();
        if (hasData) {
            const v = yMin + (yMax - yMin) * (i / yTicks);
            plotCtx.fillStyle = '#777';
            plotCtx.fillText(v.toFixed(1), PAD_L - 4, y + 3);
        }
    }

    // Y axis label
    plotCtx.fillStyle = '#555';
    plotCtx.save();
    plotCtx.translate(14, PAD_T + ph / 2);
    plotCtx.rotate(-Math.PI / 2);
    plotCtx.textAlign = 'center';
    plotCtx.fillText('Log₁₀ Loss', 0, 0);
    plotCtx.restore();

    // X axis label
    plotCtx.fillStyle = '#555';
    plotCtx.textAlign = 'center';
    plotCtx.fillText('Iteration', PAD_L + pw / 2, H - PAD_B + 18);

    // Legend: color swatches + names
    const legendY = H - PAD_B + 38;
    const totalW  = LOSS_NAMES.reduce((a, n) => a + n.length * 6 + 18, 0);
    let lx = PAD_L + (pw - totalW) / 2;
    for (let k = 0; k < 3; k++) {
        plotCtx.globalAlpha = k === activeLoss ? 1.0 : 0.25;
        plotCtx.fillStyle = LOSS_COLORS[k];
        plotCtx.fillRect(lx, legendY - 7, 10, 2);
        plotCtx.fillStyle = '#aaa';
        plotCtx.textAlign = 'left';
        plotCtx.fillText(LOSS_NAMES[k], lx + 14, legendY);
        lx += LOSS_NAMES[k].length * 6 + 24;
    }
    plotCtx.globalAlpha = 1.0;

    if (!hasData) return;

    // Draw inactive curves first (behind active)
    for (let k = 0; k < 3; k++) {
        if (k === activeLoss || !shifted[k].length) continue;
        plotCtx.globalAlpha = 0.25;
        plotCtx.strokeStyle = LOSS_COLORS[k];
        plotCtx.lineWidth = 1.5;
        plotCtx.beginPath();
        for (let i = 0; i < shifted[k].length; i++) {
            const x = PAD_L + (i / xMax) * pw;
            const y = PAD_T + ph - (shifted[k][i] - yMin) / (yMax - yMin) * ph;
            i === 0 ? plotCtx.moveTo(x, y) : plotCtx.lineTo(x, y);
        }
        plotCtx.stroke();
    }

    // Draw active curve on top
    plotCtx.globalAlpha = 1.0;
    plotCtx.strokeStyle = LOSS_COLORS[activeLoss];
    plotCtx.lineWidth = 1.5;
    plotCtx.beginPath();
    for (let i = 0; i < activeLog.length; i++) {
        const x = PAD_L + (i / xMax) * pw;
        const y = PAD_T + ph - (activeLog[i] - yMin) / (yMax - yMin) * ph;
        i === 0 ? plotCtx.moveTo(x, y) : plotCtx.lineTo(x, y);
    }
    plotCtx.stroke();
    plotCtx.globalAlpha = 1.0;
}

// ------------------------------------------------------------
// Tab switching
// ------------------------------------------------------------

const INFO_TABS = ['overview', 'instructions', 'logs'];

function switchInfoTab(tab) {
    for (const t of INFO_TABS) {
        document.getElementById(`${t}Inner`).style.display = t === tab ? '' : 'none';
        document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}`).classList.toggle('active', t === tab);
    }
    if (tab === 'logs') renderLog();
}

function switchTab(tab) {
    activeTab = tab;
    const outputInner  = document.getElementById('outputInner');
    const historyInner = document.getElementById('historyInner');
    const tabOutput    = document.getElementById('tabOutput');
    const tabHistory   = document.getElementById('tabHistory');

    if (tab === 'output') {
        outputInner.style.display  = '';
        historyInner.style.display = 'none';
        tabOutput.classList.add('active');
        tabHistory.classList.remove('active');
        // Restore current output image
        if (outputImageData) ctx.putImageData(outputImageData, 0, 0);
    } else {
        // Pause if running
        if (!isPaused && startPauseBtn.textContent === 'Pause') {
            onStartPause();
        }
        outputInner.style.display  = 'none';
        historyInner.style.display = '';
        tabOutput.classList.remove('active');
        tabHistory.classList.add('active');
        updateHistorySliders();
        renderHistoryFrame();
    }
}

function updateHistorySliders() {
    const itSlider  = document.getElementById('historyItSlider');
    const golSlider = document.getElementById('historyGolSlider');
    const itVal     = document.getElementById('historyItVal');
    const golVal    = document.getElementById('historyGolVal');

    if (historySnapshots.length === 0) {
        itSlider.disabled  = true;
        golSlider.disabled = true;
        itVal.textContent  = '—';
        golVal.textContent = '—';
        document.getElementById('historySaveCurrentBtn').disabled = true;
        document.getElementById('historySaveSeriesBtn').disabled  = true;
        return;
    }

    itSlider.min   = 0;
    itSlider.max   = historySnapshots.length - 1;
    itSlider.value = historySnapshots.length - 1;
    itSlider.disabled = false;

    const snap = historySnapshots[parseInt(itSlider.value)];
    itVal.textContent = snap.it;

    golSlider.min   = 1;
    golSlider.max   = snap.golSteps;
    golSlider.value = snap.golSteps;
    golSlider.disabled = false;
    golVal.textContent = golSlider.value;

    document.getElementById('historySaveCurrentBtn').disabled = false;
    document.getElementById('historySaveSeriesBtn').disabled  = false;
}

function onHistorySliderChange() {
    const itSlider  = document.getElementById('historyItSlider');
    const golSlider = document.getElementById('historyGolSlider');
    const itVal     = document.getElementById('historyItVal');
    const golVal    = document.getElementById('historyGolVal');

    const snap = historySnapshots[parseInt(itSlider.value)];
    if (!snap) return;

    itVal.textContent = snap.it;

    // Adjust GoL slider range to match snapshot's golSteps
    golSlider.max = snap.golSteps;
    if (parseInt(golSlider.value) > snap.golSteps) golSlider.value = snap.golSteps;
    golVal.textContent = golSlider.value;

    renderHistoryFrameDebounced();
}

let _historyRenderTimer = null;
function renderHistoryFrameDebounced() {
    clearTimeout(_historyRenderTimer);
    _historyRenderTimer = setTimeout(renderHistoryFrame, 30);
}

// Evolve a precursor grid for `steps` GoL steps and render to canvas.
// Runs on main thread (no worker) since it's interactive and fast for one image.
function renderHistoryFrame() {
    if (historySnapshots.length === 0) return;
    const itSlider  = document.getElementById('historyItSlider');
    const golSlider = document.getElementById('historyGolSlider');
    const snap = historySnapshots[parseInt(itSlider.value)];
    if (!snap) return;
    const steps = parseInt(golSlider.value);

    const evolved = evolveGrid(snap.precursorGrid, paddedW, paddedH, steps);

    // Write to outputImageData and canvas
    const imgData = ctx.createImageData(srcW, srcH);
    for (let r = 0; r < srcH; r++) {
        for (let c = 0; c < srcW; c++) {
            const v = evolved[(r + PAD) * paddedW + (c + PAD)] === 0 ? 0 : 255;
            const idx = (r * srcW + c) * 4;
            imgData.data[idx]     = v;
            imgData.data[idx + 1] = v;
            imgData.data[idx + 2] = v;
            imgData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// Pure-JS GoL step for a full padded grid (mirrors C++ logic)
function golStepJS(src, dst, W, H) {
    dst.set(src);
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            let n = 0;
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;
                    const nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < H && nc >= 0 && nc < W)
                        n += (1 - src[nr * W + nc]);
                }
            }
            const alive = 1 - src[r * W + c];
            const newAlive = (alive && (n === 2 || n === 3)) || (!alive && n === 3) ? 1 : 0;
            dst[r * W + c] = 1 - newAlive;
        }
    }
}

function evolveGrid(precursor, W, H, steps) {
    let buf0 = precursor.slice();
    let buf1 = new Uint8Array(W * H);
    for (let s = 0; s < steps; s++) {
        golStepJS(buf0, buf1, W, H);
        [buf0, buf1] = [buf1, buf0];
    }
    return buf0;
}

function onHistorySaveCurrent() {
    canvas.toBlob(blob => saveBlob(blob, 'conway_history_current.png'), 'image/png');
}

function onHistorySaveSeries() {
    const itSlider  = document.getElementById('historyItSlider');
    const snap = historySnapshots[parseInt(itSlider.value)];
    if (!snap) return;

    const N = snap.golSteps;
    const offscreen = document.createElement('canvas');
    offscreen.width  = srcW * N;
    offscreen.height = srcH;
    const offCtx = offscreen.getContext('2d');

    for (let steps = 1; steps <= N; steps++) {
        const evolved = evolveGrid(snap.precursorGrid, paddedW, paddedH, steps);
        const imgData = offCtx.createImageData(srcW, srcH);
        for (let r = 0; r < srcH; r++) {
            for (let c = 0; c < srcW; c++) {
                const v = evolved[(r + PAD) * paddedW + (c + PAD)] === 0 ? 0 : 255;
                const idx = (r * srcW + c) * 4;
                imgData.data[idx]     = v;
                imgData.data[idx + 1] = v;
                imgData.data[idx + 2] = v;
                imgData.data[idx + 3] = 255;
            }
        }
        // Draw each step side by side
        const tmp = document.createElement('canvas');
        tmp.width  = srcW;
        tmp.height = srcH;
        tmp.getContext('2d').putImageData(imgData, 0, 0);
        offCtx.drawImage(tmp, (steps - 1) * srcW, 0);
    }

    offscreen.toBlob(blob => saveBlob(blob, `conway_series_it${snap.it}.png`), 'image/png');
}

// ------------------------------------------------------------
// Loupe
// ------------------------------------------------------------

const LOUPE_SIZE    = 160;  // display size in px
const LOUPE_ZOOM    = 6;    // magnification factor
const LOUPE_OFFSET  = 16;   // gap between cursor and loupe edge

let loupeCanvas, loupeCtx;
let _lastMouseEvent = null;
let _mKeyHeld = false;

function initLoupe() {
    loupeCanvas = document.createElement('canvas');
    loupeCanvas.width  = LOUPE_SIZE;
    loupeCanvas.height = LOUPE_SIZE;
    loupeCanvas.id = 'loupeCanvas';
    loupeCanvas.style.display = 'none';
    document.body.appendChild(loupeCanvas);
    loupeCtx = loupeCanvas.getContext('2d');
    loupeCtx.imageSmoothingEnabled = false;

    canvas.addEventListener('mousemove', onLoupeMove);
    canvas.addEventListener('mouseleave', () => { loupeCanvas.style.display = 'none'; });
    document.addEventListener('keydown', e => { if ((e.key === 'm' || e.key === 'M') && !e.repeat) { _mKeyHeld = true; if (_lastMouseEvent) onLoupeMove(_lastMouseEvent); } });
    document.addEventListener('keyup',   e => { if (e.key === 'm' || e.key === 'M') { _mKeyHeld = false; loupeCanvas.style.display = 'none'; } });
}

function onLoupeMove(e) {
    _lastMouseEvent = e;
    if (!srcW || !srcH || !_mKeyHeld) {
        loupeCanvas.style.display = 'none';
        return;
    }

    // Map mouse position from display space to canvas pixel space
    const rect = canvas.getBoundingClientRect();
    const scaleX = srcW / rect.width;
    const scaleY = srcH / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top)  * scaleY;

    // Source region to sample (in canvas pixels)
    const srcRegion = LOUPE_SIZE / LOUPE_ZOOM;
    const sx = Math.round(px - srcRegion / 2);
    const sy = Math.round(py - srcRegion / 2);

    loupeCtx.fillStyle = '#000';
    loupeCtx.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
    loupeCtx.drawImage(canvas, sx, sy, srcRegion, srcRegion, 0, 0, LOUPE_SIZE, LOUPE_SIZE);

    // Position loupe near cursor, flipping to stay on screen
    const vw = window.innerWidth, vh = window.innerHeight;
    const flipX = e.clientX + LOUPE_OFFSET + LOUPE_SIZE > vw;
    const flipY = e.clientY + LOUPE_OFFSET + LOUPE_SIZE > vh;
    loupeCanvas.style.left = (flipX ? e.clientX - LOUPE_OFFSET - LOUPE_SIZE : e.clientX + LOUPE_OFFSET) + 'px';
    loupeCanvas.style.top  = (flipY ? e.clientY - LOUPE_OFFSET - LOUPE_SIZE : e.clientY + LOUPE_OFFSET) + 'px';
    loupeCanvas.style.display = 'block';
}

// ------------------------------------------------------------
// Save handlers
// ------------------------------------------------------------

function saveBlob(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
}

// ------------------------------------------------------------
// Run log — localStorage-backed persistent log
// ------------------------------------------------------------

const LOG_KEY     = 'conwayDitherLog';
const LOG_MAX     = 50;
const LOSS_FUNC_NAMES = ['Euclidean', 'Manhattan', 'Huber'];

function loadLog() {
    try {
        const entries = JSON.parse(localStorage.getItem(LOG_KEY)) || [];
        // Normalize stale in-progress entries from previous sessions
        let dirty = false;
        for (const e of entries) {
            if (e.status === 'running' || e.status === 'waiting' || e.status === 'paused') {
                if (e.id !== currentRunId) { e.status = 'complete'; dirty = true; }
            }
        }
        if (dirty) saveLog(entries);
        return entries;
    } catch { return []; }
}

function saveLog(entries) {
    try { localStorage.setItem(LOG_KEY, JSON.stringify(entries)); } catch {}
}

function logUpsert(entry) {
    const entries = loadLog();
    const idx = entries.findIndex(e => e.id === entry.id);
    if (idx >= 0) {
        entries[idx] = entry;
    } else {
        entries.push(entry);
        if (entries.length > LOG_MAX) entries.splice(0, entries.length - LOG_MAX);
    }
    saveLog(entries);
}

function updateRunLog(statusOverride) {
    if (!currentRunId) return;
    const entries = loadLog();
    const entry = entries.find(e => e.id === currentRunId);
    if (!entry) return;

    const n = allLossHistory[0].length;
    if (n > 0) {
        const l2    = allLossHistory[0][n - 1];
        const l1    = allLossHistory[1][n - 1];
        const huber = allLossHistory[2][n - 1];
        entry.losses = { l2, l1, huber };
        entry.bestLosses = {
            l2:    entry.bestLosses.l2    === null ? l2    : Math.min(entry.bestLosses.l2,    l2),
            l1:    entry.bestLosses.l1    === null ? l1    : Math.min(entry.bestLosses.l1,    l1),
            huber: entry.bestLosses.huber === null ? huber : Math.min(entry.bestLosses.huber, huber),
        };
        entry.iterations = n;
    }
    const p = getParams();
    entry.params = {
        outerIts:     p.outerIts,
        pop:          p.pop,
        tileSize:     parseInt(document.getElementById('tileSizeSlider').value),
        tileIts:      p.tileIts,
        golSteps:     p.golSteps,
        mutRateStart: p.mutRateStart,
        elitism:      p.elitism,
        revertCycles: p.revertCycles,
        decayLambda:  p.decayLambda,
        lossFunc:     p.lossFunc,
    };
    if (statusOverride) entry.status = statusOverride;
    saveLog(entries);
    renderLog();
}

function fmtLoss(v) {
    return v === null ? '—' : Math.log10(Math.max(v, 1e-6)).toFixed(2);
}

function fmtMutRate(v) {
    return v.toExponential(1);
}

function fmtRevert(v) {
    return v === 0 ? 'off' : `${v * 4} its`;
}

function fmtDate(iso) {
    const d = new Date(iso);
    const date = d.toLocaleDateString([], { month: 'numeric', day: 'numeric' });
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
    return `${date} ${time}`;
}

function renderLog(scrollToBottom = false) {
    const logsInner = document.getElementById('logsInner');
    const entries = loadLog();

    logsInner.innerHTML = '';

    // Header with clear button
    const header = document.createElement('div');
    header.id = 'logHeader';
    header.innerHTML = `<span>Run History</span>`;
    const clearBtn = document.createElement('button');
    clearBtn.id = 'clearLogBtn';
    clearBtn.textContent = 'Clear';
    clearBtn.addEventListener('click', () => {
        const entries = loadLog().filter(e => e.id === currentRunId);
        saveLog(entries);
        renderLog();
    });
    header.appendChild(clearBtn);
    logsInner.appendChild(header);

    if (entries.length === 0) {
        const empty = document.createElement('div');
        empty.style.cssText = 'padding:0.6rem 0.75rem; color:#555; font-size:0.75rem;';
        empty.textContent = 'No runs recorded yet.';
        logsInner.appendChild(empty);
        return;
    }

    for (const entry of entries) {
        const l2  = fmtLoss(entry.losses.l2);
        const l1  = fmtLoss(entry.losses.l1);
        const hub = fmtLoss(entry.losses.huber);
        const statusColor = entry.status === 'running' ? '#7af' : entry.status === 'complete' ? '#6c6' : '#888';

        const div = document.createElement('div');
        const isExpanded = expandedLogIds.has(entry.id);
        div.className = isExpanded ? 'logEntry expanded' : 'logEntry';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'logDeleteBtn';
        deleteBtn.textContent = '×';
        deleteBtn.title = 'Delete entry';
        deleteBtn.addEventListener('click', e => {
            e.stopPropagation();
            const entries = loadLog().filter(en => en.id !== entry.id);
            saveLog(entries);
            expandedLogIds.delete(entry.id);
            renderLog();
        });

        const summary = document.createElement('div');
        summary.className = 'logSummary';
        summary.innerHTML =
            `<div class="logSummaryText">` +
            `<div class="logSummaryLine1"><span class="logFilename">${entry.filename}</span></div>` +
            `<div class="logSummaryLine2">L2=${l2}  L1=${l1}  H=${hub}</div>` +
            `</div>` +
            `<div class="logSummaryRight">` +
            `<span class="logStatusRow"><span class="logStatus" style="color:${statusColor};">${entry.status}</span><span class="logChevron">${isExpanded ? 'v' : '>'}</span></span>` +
            `<span class="logDate">${fmtDate(entry.startedAt)}</span>` +
            `</div>`;

        const details = document.createElement('div');
        details.className = 'logDetails';

        const p = entry.params;
        const rows = [
            ['Population',  p.pop],
            ['Tile Size',   p.tileSize],
            ['Tile Gens',   p.tileIts],
            ['Image Its',  p.outerIts],
            ['GoL Steps',   p.golSteps],
            ['Mut. Rate',   fmtMutRate(p.mutRateStart)],
            ['Decay λ',     p.decayLambda],
            ['Revert',      fmtRevert(p.revertCycles)],
            ['Elitism',     p.elitism === 0 ? 'off' : p.elitism],
            ['Loss Func.',  LOSS_FUNC_NAMES[p.lossFunc]],
            ['Image',       `${entry.width}×${entry.height}`],
            ['Iterations',  `${entry.iterations}/${p.outerIts}`],
        ];
        for (const [k, v] of rows) {
            details.innerHTML +=
                `<span class="logKey">${k}</span><span class="logVal">${v}</span>`;
        }

        const entryBody = document.createElement('div');
        entryBody.className = 'logEntryBody';
        entryBody.appendChild(summary);
        entryBody.appendChild(details);

        div.appendChild(deleteBtn);
        div.appendChild(entryBody);

        entryBody.addEventListener('click', () => {
            const wasExpanded = div.classList.contains('expanded');
            div.classList.toggle('expanded', !wasExpanded);
            if (wasExpanded) expandedLogIds.delete(entry.id); else expandedLogIds.add(entry.id);
            summary.querySelector('.logChevron').textContent = wasExpanded ? '>' : 'v';
        });

        logsInner.appendChild(div);
    }

    if (scrollToBottom) logsInner.scrollTop = logsInner.scrollHeight;
}

function onSaveCurrent() {
    canvas.toBlob(blob => saveBlob(blob, 'conway_dither_current.png'), 'image/png');
}

function onSaveBest() {
    if (!bestImageData) return;
    const offscreen = document.createElement('canvas');
    offscreen.width  = srcW;
    offscreen.height = srcH;
    offscreen.getContext('2d').putImageData(bestImageData, 0, 0);
    offscreen.toBlob(blob => saveBlob(blob, 'conway_dither_best.png'), 'image/png');
}
