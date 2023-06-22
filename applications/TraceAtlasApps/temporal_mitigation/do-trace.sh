#!/bin/sh

ARCH="x86"

if [ "$#" -eq 1 ]; then
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

../../traceExtractCompile.sh \
  --arch ${ARCH} \
  --cpp \
  --seed-with-jr \
  --trace-compression 5 \
  --trace-home ../../../TraceAtlas/build \
  --compiler-rt ../../lib/libclang_rt.builtins-aarch64.a \
  -I../include \
  -lm \
  -d adjoint.cpp \
  -d divide.cpp \
  -d imagpart.cpp \
  -d mmult4.cpp \
  -d mmultiply.cpp \
  -d scalableinverse.cpp \
  -d alternateinverse.cpp \
  -d determinant.cpp \
  -d getcofactor.cpp \
  -d inverse.cpp \
  -d mmult64.cpp \
  -d msub.cpp \
  -d display.cpp \
  -d hermitian.cpp \
  -d mmadd.cpp \
  -d mmult.cpp \
  -d realpart.cpp \
  temporal_mitigation.cpp
