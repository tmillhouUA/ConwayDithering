#pragma once
#include <cstdint>
#include <cstddef>

extern "C" {

uint8_t* dither_alloc(size_t bytes);
void     dither_free(uint8_t* ptr);

// Optimizes a single tile's precursor state via genetic algorithm.
//
// target_gray  : float32 [tileH*tileW], grayscale values pre-scaled to [0.25, 1.0]
// precursor_in : uint8 [tileH*tileW], warm-start precursor (0=alive, 1=dead)
// precursor_out: uint8 [tileH*tileW], best precursor found
// evolved_out  : uint8 [tileH*tileW], precursor_out evolved n_its GoL steps
// returns best fitness (sum of squared differences after blur)
float dither_process_tile(
    const float*   target_gray,
    int            tileW,
    int            tileH,
    const uint8_t* precursor_in,
    uint8_t*       precursor_out,
    uint8_t*       evolved_out,
    float          mut_rate,
    int            pop,
    int            gens,
    int            n_its,
    int            blur_amt,
    int            margin,
    uint32_t       rng_seed,
    int            elitism,
    int            loss_func   // 0=L2, 1=L1, 2=Huber
);

// Computes all three losses on a pre-evolved tile (for plotting inactive functions).
// target_gray : float32 [tileH*tileW], same format as dither_process_tile
// evolved     : uint8  [tileH*tileW], the evolved output to evaluate
// losses_out  : float32[3], filled with [L2, L1, Huber]
void compute_tile_losses(
    const float*   target_gray,
    const uint8_t* evolved,
    int            tileW,
    int            tileH,
    float*         losses_out
);

} // extern "C"
