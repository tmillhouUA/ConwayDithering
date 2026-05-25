// Web Worker — loads one WASM instance and processes tile jobs.
// Stays alive across jobs; the coordinator reuses it for the entire run.

importScripts('./dither.js');

let _alloc, _free, _process, _allLosses;
let Module = null;

DitherModule().then(mod => {
    Module = mod;
    _alloc   = mod.cwrap('dither_alloc',        'number', ['number']);
    _free    = mod.cwrap('dither_free',          null,     ['number']);
    _process = mod.cwrap('dither_process_tile',  'number', [
        'number',            // target_gray ptr (float32)
        'number', 'number',  // tileW, tileH
        'number',            // precursor_in ptr (uint8)
        'number',            // precursor_out ptr (uint8)
        'number',            // evolved_out ptr (uint8)
        'number',            // mut_rate (float)
        'number',            // pop
        'number',            // gens
        'number',            // n_its
        'number',            // blur_amt
        'number',            // margin
        'number',            // rng_seed (uint32)
        'number',            // elitism (int)
        'number'             // loss_func (int): 0=L2, 1=L1, 2=Huber
    ]);
    _allLosses = mod.cwrap('compute_tile_losses', null, [
        'number',            // target_gray ptr (float32)
        'number',            // evolved ptr (uint8)
        'number', 'number',  // tileW, tileH
        'number'             // losses_out ptr (float32[3])
    ]);
    self.postMessage({ type: 'ready' });
});

self.onmessage = function(e) {
    const job = e.data;
    const tileLen = job.tileW * job.tileH;

    // Allocate WASM heap buffers
    const targetPtr  = _alloc(tileLen * 4);  // float32
    const precInPtr  = _alloc(tileLen);
    const precOutPtr = _alloc(tileLen);
    const evoOutPtr  = _alloc(tileLen);

    // Copy JS arrays into WASM heap
    Module.HEAPF32.set(job.targetTile, targetPtr >> 2);
    Module.HEAPU8.set(job.precursorIn, precInPtr);

    const fitness = _process(
        targetPtr, job.tileW, job.tileH,
        precInPtr, precOutPtr, evoOutPtr,
        job.mutRate,
        job.pop,
        job.gens,
        job.nIts,
        7,           // blur_amt
        10,          // margin
        job.seed,
        job.elitism,
        job.lossFunc
    );

    // Compute all three losses on the final evolved output (for plotting)
    const lossesPtr = _alloc(3 * 4);  // float32[3]
    _allLosses(targetPtr, evoOutPtr, job.tileW, job.tileH, lossesPtr);
    const allLosses = Array.from(Module.HEAPF32.subarray(lossesPtr >> 2, (lossesPtr >> 2) + 3));
    _free(lossesPtr);

    // Copy results out of WASM heap before freeing
    const precOut = Module.HEAPU8.slice(precOutPtr, precOutPtr + tileLen);
    const evoOut  = Module.HEAPU8.slice(evoOutPtr,  evoOutPtr  + tileLen);

    _free(targetPtr);
    _free(precInPtr);
    _free(precOutPtr);
    _free(evoOutPtr);

    // Transfer ownership of buffers (zero-copy postMessage)
    self.postMessage(
        {
            type:        'tile_done',
            tileX:       job.tileX,
            tileY:       job.tileY,
            outerIt:     job.outerIt,
            offset:      job.offset,
            evolvedOut:  evoOut,
            precursorOut: precOut,
            fitness,
            allLosses
        },
        [evoOut.buffer, precOut.buffer]
    );
};
