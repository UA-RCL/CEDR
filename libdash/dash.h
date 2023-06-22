#pragma once

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include "dash_types.h"

// List of, uh, "feature flags" that, if defined, disable the CEDR integration of their respective kernels by causing their #ifdef to only include the CPU dispatch
// This is useful for kernels that we add first to libdash but then do not yet have support for in CEDR's "enqueue_kernel" logic
#define DISABLE_BLAS_MADD_CEDR
#define DISABLE_BLAS_MSUB_CEDR
#define DISABLE_BLAS_TRANSPOSE_CEDR

#ifdef __cplusplus
extern "C" {
#endif

#define CONV_2D_MIN -32768
#define CONV_2D_MAX 32767

/*
 * Current open questions: 
 * 1. Should we be doing anything to stop the user from shooting themselves in the foot with divide-by-zero with that ZIP_DIV op?
 */
void DASH_ZIP_flt(dash_cmplx_flt_type* input_1, dash_cmplx_flt_type* input_2, dash_cmplx_flt_type* output, size_t size, zip_op_t op);
void DASH_ZIP_flt_nb(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

void DASH_ZIP_int(dash_cmplx_int_type* input_1, dash_cmplx_int_type* input_2, dash_cmplx_int_type* output, size_t size, zip_op_t op);
void DASH_ZIP_int_nb(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

/*
 * Assumes complex input and output of the form input[2*i+0] = real, input[2*i+1] = imaginary
 * "size" specifies the length of the FFT transform, so input and output should be of length 2*size
 */
void DASH_FFT_flt(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool isForwardTransform);
void DASH_FFT_flt_nb(dash_cmplx_flt_type** input, dash_cmplx_flt_type** output, size_t* size, bool* isForwardTransform, cedr_barrier_t* kernel_barrier);

void DASH_FFT_int(dash_cmplx_int_type* input, dash_cmplx_int_type* output, size_t size, bool isForwardTransform);
void DASH_FFT_int_nb(dash_cmplx_int_type** input, dash_cmplx_int_type** output, size_t* size, bool* isForwardTransform, cedr_barrier_t* kernel_barrier);

/*
 * ${Comments about usage of DASH_GEMM}
 */
void DASH_GEMM_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t Row_A, size_t Col_A, size_t Col_B);
void DASH_GEMM_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, cedr_barrier_t* kernel_barrier);

void DASH_GEMM_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t Row_A, size_t Col_A, size_t Col_B);
void DASH_GEMM_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, cedr_barrier_t* kernel_barrier);

/*
 * ${Comments about usage of DASH_BLAS_MADD}
 */ 
void DASH_BLAS_MADD_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t A_ROWS, size_t A_COLS);
void DASH_BLAS_MADD_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier);

void DASH_BLAS_MADD_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t A_ROWS, size_t A_COLS);
void DASH_BLAS_MADD_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier);

/*
 * ${Comments about usage of DASH_BLAS_MSUB}
 */ 
void DASH_BLAS_MSUB_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t A_ROWS, size_t A_COLS);
void DASH_BLAS_MSUB_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier);

void DASH_BLAS_MSUB_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t A_ROWS, size_t A_COLS);
void DASH_BLAS_MSUB_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier);

/*
 * ${Comments about usage of DASH_BLAS_TRANSPOSE}
 *
 * conjugate: if true, performs a conjugate transpose
 */ 
void DASH_BLAS_TRANSPOSE_flt(dash_cmplx_flt_type* in, dash_cmplx_flt_type* out, size_t ROWS, size_t COLS, bool conjugate);
void DASH_BLAS_TRANSPOSE_flt_nb(dash_cmplx_flt_type** in, dash_cmplx_flt_type** out, size_t* ROWS, size_t* COLS, bool* conjugate, cedr_barrier_t* kernel_barrier);

void DASH_BLAS_TRANSPOSE_int(dash_cmplx_int_type* in, dash_cmplx_int_type* out, size_t ROWS, size_t COLS, bool conjugate);
void DASH_BLAS_TRANSPOSE_int_nb(dash_cmplx_int_type** in, dash_cmplx_int_type** out, size_t* ROWS, size_t* COLS, bool* conjugate, cedr_barrier_t* kernel_barrier);

/*
 * nsymbols: number of symbols in input
 * input should be of length 2*nsymbols (in the input[2*i] = real, input[2*i+1] = imaginary format)
 */
void DASH_BPSK_flt(dash_cmplx_flt_type* input, dash_re_flt_type* output, size_t nsymbols);
void DASH_BPSK_flt_nb(dash_cmplx_flt_type** input, dash_re_flt_type** output, size_t* nsymbols, cedr_barrier_t* kernel_barrier);

void DASH_QAM16_flt(dash_cmplx_flt_type* input, int* output, size_t nsymbols);
void DASH_QAM16_flt_nb(dash_cmplx_flt_type** input, int** output, size_t* nsymbols, cedr_barrier_t* kernel_barrier);

void DASH_CONV_2D_int(dash_re_int_type *input, int height, int width, dash_re_flt_type *mask, int mask_size, dash_re_int_type *output);
void DASH_CONV_2D_int_nb(dash_re_int_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_int_type **output, cedr_barrier_t* kernel_barrier);

void DASH_CONV_1D_flt(dash_re_flt_type* input, int size, dash_re_flt_type* mask, int mask_size, dash_re_flt_type* output);
void DASH_CONV_1D_flt_nb(dash_re_flt_type** input, int* size, dash_re_flt_type** mask, int* mask_size, dash_re_flt_type** output, cedr_barrier_t* kernel_barrier);

#ifdef __cplusplus
} // Close 'extern "C"'
#endif
