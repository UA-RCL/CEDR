#!/bin/bash

ARCH="x86"

if [ "$#" -eq 1 ]; then
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

# ../../traceExtractCompile.sh \
#   --arch ${ARCH} --single-node \
#   -lm -lgsl -lgslcblas \
#   -d RF_convergence_support.c \
#   RF_convergence_LU_projection.c

#--loop-partition \
#--skip-trace \
#--skip-kernel-detection \
#../../traceExtractCompile.sh \
#  --arch ${ARCH} \
#  --inline \
#  --semantic-opt \
#  --skip-trace \
#  --skip-kernel-detection \
#  -lm -lgsl -lgslcblas \
#  -d RF_convergence_support.c \
#  RF_convergence_LU_projection.c
# --semantic-opt \

if [ "${ARCH}" == "x86" ]; then
  echo "Building for x86"
  ../../traceExtractCompile.sh \
    --arch x86 \
    --inline \
    --seed-with-jr \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -I../include \
    -lm -lgsl -lgslcblas \
    -d RF_convergence_support.c \
    RF_convergence_LU_projection.c
else
  echo "Building for aarch64"
  ../../traceExtractCompile.sh \
    --arch aarch64 \
    --inline \
    --seed-with-jr \
    --semantic-opt \
    --trace-home ../../../TraceAtlas/build \
    --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
    -I../include \
    -lm \
    -tlgsl -tlgslcblas \
    -d RF_convergence_support.c \
    -od ../../lib/libgsl-aarch64.a \
    -od ../../lib/libgslcblas-aarch64.a \
    RF_convergence_LU_projection.c
fi
