#!/usr/bin/env bash
set -e

emcc src/cpp/dither.cpp \
  -O3 \
  -msimd128 \
  --no-entry \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_dither_alloc","_dither_free","_dither_process_tile","_compute_tile_losses"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT='web,worker' \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='DitherModule' \
  -o web/dither.js

echo "Build complete. Serve the web/ directory."
