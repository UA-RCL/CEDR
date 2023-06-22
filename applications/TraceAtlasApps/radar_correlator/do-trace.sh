#!/bin/sh

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
    --trace-compression 6 \
    -I../include \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -lm \
    radar_correlator.c
elif [ "${ARCH}" == "par" ]; then
  echo "Building for x86 with parallel support"
  ../../traceExtractCompile.sh \
    --arch x86 \
    --auto-parallelize \
    -I../include \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -lm \
    radar_correlator.c
else
  echo "Building for aarch64"
  ../../traceExtractCompile.sh \
    --arch aarch64 \
    --semantic-opt \
    --seed-with-jr \
    --trace-compression 6 \
    -I../include \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -lm \
    radar_correlator.c
fi


