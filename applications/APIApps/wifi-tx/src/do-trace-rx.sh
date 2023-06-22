#!/bin/sh

ARCH="x86"

if [ "$#" -eq 1 ]; then 
  echo "Setting arch flag to $1"
  ARCH="$1"
fi

#--auto-parallelize \
#--loop-partition \
../../../traceExtractCompile.sh \
  --arch ${ARCH} \
  --loop-partition \
  --trace-compression 8 \
  --trace-home ../../../../TraceAtlas/build \
  --compiler-rt ../../../lib/libclang_rt.builtins-aarch64.a \
  -lm \
  -d crc.c \
  -d detection.c \
  -d baseband_lib.c \
  -d viterbi.c \
  -d interleaver_deintleaver.c \
  -d scrambler_descrambler.c \
  -d qpsk_Mod_Demod.c \
  -d CyclicPrefix.c \
  -d Preamble_ST_LG.c \
  -d channel_Eq.c \
  -d channel_Eq_initialize.c \
  -d channel_Eq_terminate.c \
  -d fft.c \
  -d fft_hwa.c \
  -d dma.c \
  -d diag.c \
  -d rdivide.c \
  -d rtGetInf.c \
  -d rtGetNaN.c \
  -d rt_nonfinite.c \
  -d decode.c \
  -d datatypeconv.c \
  -d fft_hs.c \
  -d rfInf.c \
  -d pilot.c \
  -d equalizer.c \
  RX.c
