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

// ------------------------------------------------------------
// Entry point — called from index.html after DOM ready
// ------------------------------------------------------------

function init() {
    canvas      = document.getElementById('canvas');
    ctx         = canvas.getContext('2d');
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
    paddedW = srcW + 2 * PAD;
    paddedH = srcH + 2 * PAD;
    ySteps = Math.floor(srcH / STEP);
    xSteps = Math.floor(srcW / STEP);

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

    document.getElementById('fileInput').disabled = true;
    downloadBtn.disabled = true;
    uploadArea.style.display = 'none';
    canvas.style.display = 'block';

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
    let mutRate = 0.001;
    for (let outerIt = 0; outerIt < OUTER_ITS; outerIt++) {
        if (outerIt % 50 === 0) mutRate /= 2;
        await runOneIteration(outerIt, mutRate);
        const pct = Math.round((outerIt + 1) / OUTER_ITS * 100);
        progressEl.textContent = `Iteration ${outerIt + 1} / ${OUTER_ITS} — ${pct}%`;
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

        pendingTiles--;
        if (pendingTiles === 0) {
            for (const w of pendingPrecursorWrites)
                writeTileToPrecursorGrid(w.tileX, w.tileY, w.offset, w.precursorOut);
            pinBorder(precursorGrid, paddedW, paddedH, PAD);
            ctx.putImageData(outputImageData, 0, 0);
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
