#!/bin/sh

ARCH="x86"

if [ "$#" -eq 1 ]; then
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

if [ "${ARCH}" == "x86" ]; then
  echo "Building for x86"
  ../../../traceExtractCompile.sh \
    --arch x86 -lm -lgsl -lgslcblas --loop-partition --trace-compression 8 \
    pulse_doppler.c
else
  #-od /home/jmack2545/Downloads/RADAR/ARM/lib/libgsl.a \
  #-od /home/jmack2545/Downloads/RADAR/ARM/lib/libgslcblas.a \
  echo "Building for aarch64"
  ../../../traceExtractCompile.sh \
    --arch aarch64 -tlgsl -tlgslcblas -lm --loop-partition --trace-compression 8 \
    -od /localhome/jmack2545/local-libs/gsl-arm64-docker/libgsl.a \
    -od /localhome/jmack2545/local-libs/gsl-arm64-docker/libgslcblas.a \
    pulse_doppler.c
fi
