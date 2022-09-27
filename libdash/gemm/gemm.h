#pragma once

#include "dash_types.h"
#include "platform.h"

#if DASH_PLATFORM == DASH_JETSONAGX
  #error "Current platform does not support the GEMM module. You likely want the GEMM implementation of the GPU module instead"
#endif

// What is the baseline data type that the accelerator works with?
// We need to convert whatever baseline type we're given into this
typedef float gemm_re_type;

// And then we pack that data type together into re/im complex inputs
typedef struct gemm_cmplx_type {
  gemm_re_type re;
  gemm_re_type im;
} gemm_cmplx_type;


#define NUM_GEMMS (sizeof(GEMM_DMA_CTRL_BASE_ADDRS) / sizeof(GEMM_DMA_CTRL_BASE_ADDRS[0]))

// 64x4 * 4x64 matrices with re/im parts
#define A_ROWS      4
#define A_COLS      64
#define B_ROWS      (A_COLS)
#define B_COLS      4

#define C_ROWS      (A_ROWS)
#define C_COLS      (B_COLS)

#define A_SIZE      (A_ROWS * A_COLS * 2)
#define B_SIZE      (B_ROWS * B_COLS * 2)
#define C_SIZE      (C_ROWS * C_COLS * 2)

#define INPUT_DIM   (A_SIZE + B_SIZE)
#define OUTPUT_DIM  (C_SIZE)

#define UDMABUF_PARTITION_SIZE (GEMM_UDMABUF_SIZE / NUM_GEMMS)
// Make sure that our udmabuf partitions are sized such that we can hold INPUT + OUTPUT for each GEMM in non-conflicting buffers
#define REQUIRED_BUFFER_SIZE ((INPUT_DIM + OUTPUT_DIM) * sizeof(float))
static_assert(UDMABUF_PARTITION_SIZE >= REQUIRED_BUFFER_SIZE, "Current udmabuf size is too small to support this many GEMMs!");

#define __DASH_GEMM_DEBUG__

#ifdef LOG
#undef LOG
#endif

#ifdef __DASH_GEMM_DEBUG__
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)

#endif
