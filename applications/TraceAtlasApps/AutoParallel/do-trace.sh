#!/bin/sh

ARCH="x86"

if [ "$#" -eq 1 ]; then
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

../../traceExtractCompile.sh \
  --arch ${ARCH} \
  --auto-parallelize \
  --trace-home ../../../TraceAtlas/build \
  parallel.c
