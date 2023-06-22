#include "dash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(CPU_ONLY)
extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);
#endif

void DASH_GEMM_flt_cpu(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS) {
  dash_re_flt_type res1, res2, res3, res4;
  dash_re_flt_type term1, term2, term3, term4;

  for (int i = 0; i < (*A_ROWS); i++) {
    for (int j = 0; j < (*B_COLS); j++) {
      res1 = 0; res2 = 0; res3 = 0; res4 = 0;
      // A_COLS better equal B_ROWS, I'm trusting you here >:(
      for (int k = 0; k < (*A_COLS); k++) {
        term1 = (*A)[i * (*A_COLS) + k].re * (*B)[k * (*B_COLS) + j].re;
        res1 += term1;

        term2 = (*A)[i * (*A_COLS) + k].im * (*B)[k * (*B_COLS) + j].im;
        res2 += term2;

        term3 = (*A)[i * (*A_COLS) + k].re * (*B)[k * (*B_COLS) + j].im;
        res3 += term3;

        term4 = (*A)[i * (*A_COLS) + k].im * (*B)[k * (*B_COLS) + j].re;
        res4 += term4;
      }
    (*C)[i * (*B_COLS) + j].re = res1 - res2;
    (*C)[i * (*B_COLS) + j].im = res3 + res4;
    }
  }
}

void DASH_GEMM_int_cpu(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS) {
  dash_re_int_type res1, res2, res3, res4;
  dash_re_int_type term1, term2, term3, term4;

  for (int i = 0; i < (*A_ROWS); i++) {
    for (int j = 0; j < (*B_COLS); j++) {
      res1 = 0; res2 = 0; res3 = 0; res4 = 0;
      // A_COLS better equal B_ROWS, I'm trusting you here >:(
      for (int k = 0; k < (*A_COLS); k++) {
        term1 = (*A)[i * (*A_COLS) + k].re * (*B)[k * (*B_COLS) + j].re;
        res1 += term1;

        term2 = (*A)[i * (*A_COLS) + k].im * (*B)[k * (*B_COLS) + j].im;
        res2 += term2;

        term3 = (*A)[i * (*A_COLS) + k].re * (*B)[k * (*B_COLS) + j].im;
        res3 += term3;

        term4 = (*A)[i * (*A_COLS) + k].im * (*B)[k * (*B_COLS) + j].re;
        res4 += term4;
      }
    (*C)[i * (*B_COLS) + j].re = res1 - res2;
    (*C)[i * (*B_COLS) + j].im = res3 + res4;
    }
  }
}

void DASH_GEMM_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_GEMM_CEDR)
  DASH_GEMM_flt_cpu(A, B, C, A_ROWS, A_COLS, B_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_GEMM", "flt", 7, A, B, C, A_ROWS, A_COLS, B_COLS, kernel_barrier);
#endif
}

void DASH_GEMM_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t A_ROWS, size_t A_COLS, size_t B_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_GEMM_CEDR)
  DASH_GEMM_flt_cpu(&A, &B, &C, &A_ROWS, &A_COLS, &B_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_GEMM_flt_nb(&A, &B, &C, &A_ROWS, &A_COLS, &B_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_GEMM_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_GEMM_CEDR)
  DASH_GEMM_int_cpu(A, B, C, A_ROWS, A_COLS, B_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_GEMM", "int", 7, A, B, C, A_ROWS, A_COLS, B_COLS, kernel_barrier);
#endif
}

void DASH_GEMM_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t A_ROWS, size_t A_COLS, size_t B_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_GEMM_CEDR)
  DASH_GEMM_int_cpu(&A, &B, &C, &A_ROWS, &A_COLS, &B_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_GEMM_int_nb(&A, &B, &C, &A_ROWS, &A_COLS, &B_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
} // Close 'extern "C"'
#endif