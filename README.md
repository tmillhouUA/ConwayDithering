# Conway Dithering

A web tool that converts grayscale images into one-bit Conway's Game of Life patterns. The output is not merely styled to look like the Game of Life — it is produced by running the Life rules forward from an evolutionarily-optimized starting state. I originally developed this tool in Python, and used it to create the cover of my co-edited volume, [*Dennett's Real Patterns in Science and Nature*](https://mitpress.mit.edu/9780262052030/dennetts-real-patterns-in-science-and-nature/) (MIT Press, 2026). I re-wrote the tool in C++ and built the web interface with the assistance of Claude Code. 

## How It Works

The image is divided into overlapping tiles. For each tile, a genetic algorithm searches for a starting configuration (the *precursor*) that, after a fixed number of Game of Life steps, approximates the target image. Candidates are mutated and evaluated over many generations, with the mutation rate decaying as the image converges. Scoring is done on a blurred version of both the evolved output and the target, which shifts the objective from pixel-level matching to density matching — producing the right proportion of live cells in each local area rather than a hard threshold.

## Usage

A live version is available at [tmillhouua.github.io/ConwayDithering](https://tmillhouua.github.io/ConwayDithering/).

Upload any image (or use the built-in test image) to begin. I advise scaling your chosen image to approximately 400x600 pixels. Larger images will take longer to evolve and will make the Life pattern hard to discern at normal viewing sizes. Alternatively, use the "Scale to fit screen" checkbox to automatically scale your image to a reasonable size. I also recommend an image whose subject takes up most of the frame and contrasts sharply with the background (e.g., a portrait of a person against a white background). Full descriptions of all controls and parameters are available in the **How To** tab inside the tool. Background on the algorithm is in the **About** tab.

## Implementation

The core algorithm is written in C++ and compiled to WebAssembly with Emscripten. Tile jobs run in parallel across a pool of Web Workers, one WASM instance per worker. The main thread coordinates job dispatch, renders the canvas tile-by-tile as results arrive, and manages the evolutionary state across iterations.

| Component | Description |
|---|---|
| `src/cpp/dither.cpp` | C++ core: genetic algorithm, GoL step, Gaussian blur, fitness scoring |
| `docs/coordinator.js` | Main thread: worker pool, canvas rendering, UI, run log |
| `docs/worker.js` | Web Worker: loads WASM, executes tile jobs |
| `docs/sw.js` | Service worker: injects COOP/COEP headers required for `SharedArrayBuffer` |
| `build.sh` | Emscripten compile script: builds `dither.cpp` into `dither.js` and `dither.wasm` |

## Building

Requires [Emscripten](https://emscripten.org/). With `emsdk` activated:

```bash
bash build.sh
```

Output (`dither.js`, `dither.wasm`) is written directly to `docs/`. Serve `docs/` locally with any server that sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` — the included service worker handles this automatically when served over HTTPS or localhost.

## Dependencies

No runtime JavaScript dependencies. The C++ core uses only the standard library.
