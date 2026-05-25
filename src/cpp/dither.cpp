#include "dither.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>

// ------------------------------------------------------------
// Memory helpers
// ------------------------------------------------------------

uint8_t* dither_alloc(size_t bytes) {
    return static_cast<uint8_t*>(malloc(bytes));
}

void dither_free(uint8_t* ptr) {
    free(ptr);
}

// ------------------------------------------------------------
// xorshift32 — fast, seedable, thread-safe RNG
// ------------------------------------------------------------

static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Returns float in [0, 1)
static float rng_float(uint32_t& state) {
    return (xorshift32(state) & 0x00FFFFFFu) / float(0x01000000u);
}

// ------------------------------------------------------------
// Conway's Game of Life — single step
// Convention: 0 = alive, 1 = dead
// Zero-padded boundaries (cells outside grid treated as dead=1, alive=0)
// ------------------------------------------------------------

static void gol_step(const uint8_t* __restrict__ in,
                           uint8_t* __restrict__ out,
                     int W, int H) {
    // Interior: no bounds checks — lets compiler auto-vectorize with SIMD
    for (int r = 1; r < H - 1; r++) {
        const uint8_t* row0 = in + (r - 1) * W;
        const uint8_t* row1 = in +  r      * W;
        const uint8_t* row2 = in + (r + 1) * W;
        for (int c = 1; c < W - 1; c++) {
            int n = (1-row0[c-1]) + (1-row0[c]) + (1-row0[c+1])
                  + (1-row1[c-1])               + (1-row1[c+1])
                  + (1-row2[c-1]) + (1-row2[c]) + (1-row2[c+1]);
            int alive = 1 - row1[c];
            int new_alive = (alive & ((n == 2) | (n == 3))) | ((!alive) & (n == 3));
            out[r * W + c] = static_cast<uint8_t>(1 - new_alive);
        }
    }
    // Border: bounds-checked, ~6% of cells
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            if (r > 0 && r < H - 1 && c > 0 && c < W - 1) continue;
            int n = 0;
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < H && nc >= 0 && nc < W)
                        n += (1 - in[nr * W + nc]);
                }
            }
            int alive = 1 - in[r * W + c];
            int new_alive = (alive && (n == 2 || n == 3)) || (!alive && n == 3);
            out[r * W + c] = static_cast<uint8_t>(1 - new_alive);
        }
    }
}

// ------------------------------------------------------------
// Gaussian blur — separable, matches OpenCV GaussianBlur(ksize=7, sigma=0)
// sigma = 0.3*(3-1) + 0.8 = 1.4
// Kernel computed for x in [-3..3]: exp(-x^2 / (2*1.4^2)), normalized
// ------------------------------------------------------------

// OpenCV getGaussianKernel(7, 0) — binomial approximation [1,6,15,20,15,6,1]/64
static const float GAUSS7[7] = {
    0.03125f, 0.109375f, 0.21875f, 0.28125f,
    0.21875f, 0.109375f, 0.03125f
};

// Reflect index — matches OpenCV BORDER_REFLECT_101 (default for GaussianBlur)
// e.g. for max=5: ..., 2, 1, 0, 1, 2, 3, 4, 3, 2, ...
static inline int reflect101_idx(int i, int max) {
    if (max == 1) return 0;
    while (i < 0 || i >= max) {
        if (i < 0)    i = -i;
        if (i >= max) i = 2 * (max - 1) - i;
    }
    return i;
}

static void gaussian_blur(const float* __restrict__ in,
                                float* __restrict__ tmp,
                                float* __restrict__ out,
                           int W, int H) {
    // Horizontal pass: in -> tmp (reflect-101)
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            float sum = 0.0f;
            for (int k = -3; k <= 3; k++)
                sum += GAUSS7[k + 3] * in[r * W + reflect101_idx(c + k, W)];
            tmp[r * W + c] = sum;
        }
    }
    // Vertical pass: tmp -> out (reflect-101)
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            float sum = 0.0f;
            for (int k = -3; k <= 3; k++)
                sum += GAUSS7[k + 3] * tmp[reflect101_idx(r + k, H) * W + c];
            out[r * W + c] = sum;
        }
    }
}

// ------------------------------------------------------------
// Loss functions: 0=L2 (Euclidean), 1=L1 (Manhattan), 2=Huber
// ------------------------------------------------------------

static const float HUBER_DELTA = 0.15f;

static float compute_loss(const float* __restrict__ a,
                          const float* __restrict__ b,
                          int n, int loss_func) {
    float sum = 0.0f;
    if (loss_func == 1) {
        for (int i = 0; i < n; i++)
            sum += fabsf(a[i] - b[i]);
    } else if (loss_func == 2) {
        for (int i = 0; i < n; i++) {
            float d = fabsf(a[i] - b[i]);
            sum += (d <= HUBER_DELTA)
                ? 0.5f * d * d
                : HUBER_DELTA * (d - 0.5f * HUBER_DELTA);
        }
    } else {
        for (int i = 0; i < n; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
    }
    return sum;
}

// ------------------------------------------------------------
// All-losses evaluator (for plotting inactive loss functions)
// ------------------------------------------------------------

void compute_tile_losses(
    const float*   target_gray,
    const uint8_t* evolved,
    int            tileW,
    int            tileH,
    float*         losses_out)
{
    const int N = tileW * tileH;
    float* blur_in  = static_cast<float*>(malloc(N * sizeof(float)));
    float* blur_tmp = static_cast<float*>(malloc(N * sizeof(float)));
    float* blur_out = static_cast<float*>(malloc(N * sizeof(float)));
    float* tgt_tmp  = static_cast<float*>(malloc(N * sizeof(float)));
    float* tgt_blur = static_cast<float*>(malloc(N * sizeof(float)));

    for (int i = 0; i < N; i++)
        blur_in[i] = static_cast<float>(evolved[i]);
    gaussian_blur(blur_in, blur_tmp, blur_out, tileW, tileH);
    gaussian_blur(target_gray, tgt_tmp, tgt_blur, tileW, tileH);

    losses_out[0] = compute_loss(tgt_blur, blur_out, N, 0);
    losses_out[1] = compute_loss(tgt_blur, blur_out, N, 1);
    losses_out[2] = compute_loss(tgt_blur, blur_out, N, 2);

    free(blur_in); free(blur_tmp); free(blur_out);
    free(tgt_tmp); free(tgt_blur);
}

// ------------------------------------------------------------
// Main tile processor
// ------------------------------------------------------------

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
    int            loss_func)
{
    const int N = tileW * tileH;

    // --- Allocate working buffers ---
    // Population: pop precursor grids
    uint8_t* population = static_cast<uint8_t*>(malloc(pop * N));
    // GoL evolution scratch buffers (two uint8 bufs, ping-pong)
    uint8_t* gol_buf0   = static_cast<uint8_t*>(malloc(N));
    uint8_t* gol_buf1   = static_cast<uint8_t*>(malloc(N));
    // Float buffers for blur
    float* blur_in      = static_cast<float*>(malloc(N * sizeof(float)));
    float* blur_tmp     = static_cast<float*>(malloc(N * sizeof(float)));
    float* blur_out     = static_cast<float*>(malloc(N * sizeof(float)));
    // Blurred target (computed once per tile, same boundary as evolved blur)
    float* blurred_target = static_cast<float*>(malloc(N * sizeof(float)));
    float* target_tmp     = static_cast<float*>(malloc(N * sizeof(float)));
    gaussian_blur(target_gray, target_tmp, blurred_target, tileW, tileH);
    free(target_tmp);
    // Fitness array
    float* fitness      = static_cast<float*>(malloc(pop * sizeof(float)));

    // Initialize population from precursor_in (all clones)
    for (int j = 0; j < pop; j++)
        memcpy(population + j * N, precursor_in, N);

    uint32_t rng = rng_seed ? rng_seed : 1u;
    float best_fitness = FLT_MAX;
    int   best_idx = 0;

    for (int gen = 0; gen < gens; gen++) {

        // --- Mutate non-elite individuals ---
        // Slots [0, elitism) are elite (unmutated); rest are mutated
        for (int j = elitism; j < pop; j++) {
            uint8_t* indiv = population + j * N;
            for (int r = 0; r < tileH; r++) {
                if (r < margin || r >= tileH - margin) continue;
                for (int c = 0; c < tileW; c++) {
                    if (c < margin || c >= tileW - margin) continue;
                    if (rng_float(rng) < mut_rate)
                        indiv[r * tileW + c] ^= 1;
                }
            }
        }

        // --- Evaluate all individuals ---
        best_fitness = FLT_MAX;
        for (int j = 0; j < pop; j++) {
            uint8_t* indiv = population + j * N;

            // Evolve n_its GoL steps, ping-ponging between gol_buf0 and gol_buf1
            memcpy(gol_buf0, indiv, N);
            for (int step = 0; step < n_its; step++) {
                uint8_t* src = (step % 2 == 0) ? gol_buf0 : gol_buf1;
                uint8_t* dst = (step % 2 == 0) ? gol_buf1 : gol_buf0;
                gol_step(src, dst, tileW, tileH);
            }
            uint8_t* evolved = (n_its % 2 == 0) ? gol_buf0 : gol_buf1;

            // Convert uint8 to float for blur
            for (int i = 0; i < N; i++)
                blur_in[i] = static_cast<float>(evolved[i]);  // 0.0 or 1.0

            // Gaussian blur the evolved grid
            gaussian_blur(blur_in, blur_tmp, blur_out, tileW, tileH);

            fitness[j] = compute_loss(blurred_target, blur_out, N, loss_func);

            if (fitness[j] < best_fitness) {
                best_fitness = fitness[j];
                best_idx = j;
            }
        }

        // --- Clone best: fill elite slots first, then rest ---
        uint8_t* best_precursor = population + best_idx * N;
        // Compact: move best to slot 0, fill remaining slots as copies
        if (best_idx != 0)
            memcpy(population, best_precursor, N);
        for (int j = 1; j < pop; j++)
            memcpy(population + j * N, population, N);
        best_idx = 0;

        if (best_fitness < 1.0f) break;
    }

    // --- Write outputs ---
    uint8_t* best_precursor = population + best_idx * N;
    memcpy(precursor_out, best_precursor, N);

    // Evolve best precursor n_its steps to get evolved_out
    memcpy(gol_buf0, best_precursor, N);
    for (int step = 0; step < n_its; step++) {
        uint8_t* src = (step % 2 == 0) ? gol_buf0 : gol_buf1;
        uint8_t* dst = (step % 2 == 0) ? gol_buf1 : gol_buf0;
        gol_step(src, dst, tileW, tileH);
    }
    uint8_t* final_evolved = (n_its % 2 == 0) ? gol_buf0 : gol_buf1;
    memcpy(evolved_out, final_evolved, N);

    // --- Cleanup ---
    free(population);
    free(gol_buf0);
    free(gol_buf1);
    free(blur_in);
    free(blur_tmp);
    free(blur_out);
    free(blurred_target);
    free(fitness);

    return best_fitness;
}
