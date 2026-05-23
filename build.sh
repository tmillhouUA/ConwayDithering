#!/usr/bin/env bash
set -e

mkdir -p dist

emcc src/cpp/dither.cpp \
  -O3 \
  -msimd128 \
  --no-entry \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_dither_alloc","_dither_free","_dither_process_tile"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT='web,worker' \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='DitherModule' \
  -o dist/dither.js

cp web/* dist/

echo "Build complete. Serve the dist/ directory to test."
