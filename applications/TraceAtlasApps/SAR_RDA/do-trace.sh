#!/bin/bash

ARCH="x86"

if [ "$#" -eq 1 ]; then
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

if [ "${ARCH}" == "x86" ]; then
  echo "Building for x86"
  ../../traceExtractCompile.sh \
    --arch x86 \
    --seed-with-jr \
    --trace-home ../../../TraceAtlas/build \
    -I../include \
    -lm -lgsl -lgslcblas \
    SAR_RDA.c
else
  ../../traceExtractCompile.sh \
    --arch aarch64 \
    --seed-with-jr \
    --semantic-opt \
    --debug \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -I../include \
    -lm \
    -tlgsl -tlgslcblas \
    -od ../../lib/libgsl.a \
    -od ../../lib/libgslcblas.a \
    SAR_RDA.c
fi
