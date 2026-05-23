// Main thread coordinator — manages the worker pool, outer GA loop,
// canvas rendering, and progress reporting.

const STEP      = 50;
const HALF_STEP = 25;
const MARGIN    = 6;
const OFFSETS   = [0, 25, 13, 38];    // 0, 1/2, 1/4, 3/4 of STEP (rounded)
const PAD       = MARGIN + Math.max(...OFFSETS);  // 44 — enough for largest offset
const TILE_SIZE = STEP + 2 * MARGIN;  // 62
const OUTER_ITS = 500;
const MAX_WORKERS = Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) * 0.75));

let workers    = [];
let workerBusy = [];
let jobQueue   = [];
let pendingTiles = 0;
let outerItResolver = null;
let pendingPrecursorWrites = [];

let srcW, srcH, paddedW, paddedH;
let ySteps, xSteps;
let precursorGrid;   // Uint8Array — full padded precursor state
let paddedTarget;    // Float32Array — full padded target in [0.25, 1.0]
let outputImageData; // ImageData written to canvas
let canvas, ctx, progressEl, downloadBtn, uploadArea;
let plotCanvas, plotCtx;
let fitnessHistory = [];   // avg fitness per outer iteration
let tileFitnessSum, tileFitnessCount;

// ------------------------------------------------------------
// Entry point — called from index.html after DOM ready
// ------------------------------------------------------------

function init() {
    canvas      = document.getElementById('canvas');
    ctx         = canvas.getContext('2d');
    plotCanvas  = document.getElementById('plot');
    plotCtx     = plotCanvas.getContext('2d');
    progressEl  = document.getElementById('progress');
    downloadBtn = document.getElementById('downloadBtn');
    uploadArea  = document.getElementById('uploadArea');

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
    downloadBtn.addEventListener('click', onDownload);

    spawnWorkers();
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
        document.getElementById('statusText').textContent = 'Ready. Upload an image to begin.';
    }
}

// ------------------------------------------------------------
// File loading
// ------------------------------------------------------------

function onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => prepareAndRun(img);
    img.src = URL.createObjectURL(file);
}

function prepareAndRun(img) {
    const offscreen = document.createElement('canvas');
    offscreen.width  = img.naturalWidth;
    offscreen.height = img.naturalHeight;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(img, 0, 0);
    const imageData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);

    srcW = offscreen.width;
    srcH = offscreen.height;
    ySteps = Math.ceil(srcH / STEP);
    xSteps = Math.ceil(srcW / STEP);
    // Pad based on tile coverage, not raw src size, so edge tiles don't read past the buffer
    paddedW = xSteps * STEP + 2 * PAD;
    paddedH = ySteps * STEP + 2 * PAD;

    // Build padded float target [0.25, 1.0], 1.0 = dead = white border
    paddedTarget = new Float32Array(paddedH * paddedW).fill(1.0);
    for (let r = 0; r < srcH; r++) {
        for (let c = 0; c < srcW; c++) {
            const px = imageData.data[(r * srcW + c) * 4];  // red channel (grayscale)
            const v  = (px / 255.0) * 0.75 + 0.25;
            paddedTarget[(r + PAD) * paddedW + (c + PAD)] = v;
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
    downloadBtn.disabled = true;
    uploadArea.style.display = 'none';
    document.getElementById('viewer').style.display = 'flex';
    fitnessHistory = [];
    drawPlot();

    runAllIterations();
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
    const MUT_RATE_START = 0.001;
    const MUT_RATE_DECAY = Math.pow(2, -10 / OUTER_ITS);  // same total decay as halving 10x
    for (let outerIt = 0; outerIt < OUTER_ITS; outerIt++) {
        const mutRate = MUT_RATE_START * Math.pow(MUT_RATE_DECAY, outerIt);
        await runOneIteration(outerIt, mutRate);
        const pct = Math.round((outerIt + 1) / OUTER_ITS * 100);
        const avgFitness = fitnessHistory[fitnessHistory.length - 1];
        progressEl.textContent = `Iteration ${outerIt + 1} / ${OUTER_ITS} — ${pct}% — Log Fitness: ${Math.log10(Math.max(avgFitness, 1e-6)).toFixed(2)}`;
    }
    downloadBtn.disabled = false;
    progressEl.textContent = 'Done!';
}

function runOneIteration(outerIt, mutRate) {
    return new Promise(resolve => {
        outerItResolver = resolve;

        // Cycle through 4 tile grid offsets each iteration
        const offset = OFFSETS[outerIt % 4];

        pendingTiles = ySteps * xSteps;
        pendingPrecursorWrites = [];
        jobQueue = [];
        tileFitnessSum = 0;
        tileFitnessCount = 0;

        for (let ty = 0; ty < ySteps; ty++) {
            for (let tx = 0; tx < xSteps; tx++) {
                const targetTile  = extractTileF32(paddedTarget,  tx, ty, offset);
                const precursorIn = extractTileU8 (precursorGrid, tx, ty, offset);
                jobQueue.push({
                    outerIt, tileX: tx, tileY: ty, offset,
                    targetTile, precursorIn,
                    tileW: TILE_SIZE, tileH: TILE_SIZE,
                    mutRate,
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

        pendingTiles--;
        if (pendingTiles === 0) {
            fitnessHistory.push(tileFitnessSum / tileFitnessCount);
            for (const w of pendingPrecursorWrites)
                writeTileToPrecursorGrid(w.tileX, w.tileY, w.offset, w.precursorOut);
            pinBorder(precursorGrid, paddedW, paddedH, PAD);
            ctx.putImageData(outputImageData, 0, 0);
            drawPlot();
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
    // Vertical budget: viewport minus header (~60px), controls (~60px), padding and gaps (~40px)
    const maxH = Math.max(200, window.innerHeight - 160);
    const maxW = window.innerWidth * 0.6;  // 3/5ths of screen width
    const scale = Math.min(maxW / srcW, maxH / srcH);
    canvas.style.width  = Math.round(srcW * scale) + 'px';
    canvas.style.height = Math.round(srcH * scale) + 'px';
}

// ------------------------------------------------------------
// Fitness plot
// ------------------------------------------------------------

function drawPlot() {
    const W = plotCanvas.width;
    const H = plotCanvas.height;
    const PAD_L = 42, PAD_R = 10, PAD_T = 10, PAD_B = 30;
    const pw = W - PAD_L - PAD_R;
    const ph = H - PAD_T - PAD_B;

    plotCtx.fillStyle = '#111';
    plotCtx.fillRect(0, 0, W, H);

    if (fitnessHistory.length === 0) return;

    const window = 10;
    const smoothed = fitnessHistory.map((_, i) => {
        const slice = fitnessHistory.slice(Math.max(0, i - window + 1), i + 1);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
    const logVals = smoothed.map(v => Math.log10(Math.max(v, 1e-6)));
    const dataMin = Math.min(...logVals);
    const dataMax = Math.max(...logVals);
    const dataRange = Math.max(dataMax - dataMin, 0.1);
    const margin = dataRange * 0.125;  // 4/5ths of axis used by data
    const yMin = dataMin - margin;
    const yMax = dataMax + margin;
    const xMax = fitnessHistory.length / 0.9;

    // Grid lines + Y axis labels
    plotCtx.strokeStyle = '#333';
    plotCtx.fillStyle = '#777';
    plotCtx.font = '10px monospace';
    plotCtx.textAlign = 'right';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
        const v = yMin + (yMax - yMin) * i / yTicks;
        const y = PAD_T + ph - (v - yMin) / (yMax - yMin) * ph;
        plotCtx.beginPath();
        plotCtx.moveTo(PAD_L, y);
        plotCtx.lineTo(PAD_L + pw, y);
        plotCtx.stroke();
        plotCtx.fillText(v.toFixed(1), PAD_L - 4, y + 3);
    }

    // X axis label
    plotCtx.fillStyle = '#555';
    plotCtx.textAlign = 'center';
    plotCtx.fillText('iteration', PAD_L + pw / 2, H - 4);

    // Y axis label
    plotCtx.save();
    plotCtx.translate(10, PAD_T + ph / 2);
    plotCtx.rotate(-Math.PI / 2);
    plotCtx.fillText('log₁₀ fitness', 0, 0);
    plotCtx.restore();

    // Plot line
    plotCtx.strokeStyle = '#7af';
    plotCtx.lineWidth = 1.5;
    plotCtx.beginPath();
    for (let i = 0; i < logVals.length; i++) {
        const x = PAD_L + (i / xMax) * pw;
        const y = PAD_T + ph - (logVals[i] - yMin) / (yMax - yMin) * ph;
        i === 0 ? plotCtx.moveTo(x, y) : plotCtx.lineTo(x, y);
    }
    plotCtx.stroke();
}

// ------------------------------------------------------------
// Download
// ------------------------------------------------------------

function onDownload() {
    canvas.toBlob(blob => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'conway_dither.png';
        a.click();
    }, 'image/png');
}
