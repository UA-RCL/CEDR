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

void DASH_BLAS_MADD_flt_cpu(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS) {
  for (size_t i = 0; i < (*A_ROWS); i++) {
    for (size_t j = 0; j < (*A_COLS); j++) {
      (*C)[(i * (*A_COLS)) + j].re = (*A)[(i * (*A_COLS)) + j].re + (*B)[(i * (*A_COLS)) + j].re;
      (*C)[(i * (*A_COLS)) + j].im = (*A)[(i * (*A_COLS)) + j].im + (*B)[(i * (*A_COLS)) + j].im;
    }
  }
}

void DASH_BLAS_MADD_int_cpu(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS) {
  for (size_t i = 0; i < (*A_ROWS); i++) {
    for (size_t j = 0; j < (*A_COLS); j++) {
      (*C)[(i * (*A_COLS)) + j].re = (*A)[(i * (*A_COLS)) + j].re + (*B)[(i * (*A_COLS)) + j].re;
      (*C)[(i * (*A_COLS)) + j].im = (*A)[(i * (*A_COLS)) + j].im + (*B)[(i * (*A_COLS)) + j].im;
    }
  }
}

void DASH_BLAS_MSUB_flt_cpu(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS) {
  for (size_t i = 0; i < (*A_ROWS); i++) {
    for (size_t j = 0; j < (*A_COLS); j++) {
      (*C)[(i * (*A_COLS)) + j].re = (*A)[(i * (*A_COLS)) + j].re - (*B)[(i * (*A_COLS)) + j].re;
      (*C)[(i * (*A_COLS)) + j].im = (*A)[(i * (*A_COLS)) + j].im - (*B)[(i * (*A_COLS)) + j].im;
    }
  }
}

void DASH_BLAS_MSUB_int_cpu(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS) {
  for (size_t i = 0; i < (*A_ROWS); i++) {
    for (size_t j = 0; j < (*A_COLS); j++) {
      (*C)[(i * (*A_COLS)) + j].re = (*A)[(i * (*A_COLS)) + j].re - (*B)[(i * (*A_COLS)) + j].re;
      (*C)[(i * (*A_COLS)) + j].im = (*A)[(i * (*A_COLS)) + j].im - (*B)[(i * (*A_COLS)) + j].im;
    }
  }
}

void DASH_BLAS_TRANSPOSE_flt_cpu(dash_cmplx_flt_type** in, dash_cmplx_flt_type** out, size_t* ROWS, size_t* COLS, bool* conjugate) {
  // Input is ROWS x COLS, so each "row" needs to skip COLS elements ahead and there are ROWS many of them
  // Output is COLS x ROWS, so each "row" needs to skip ROWS elements ahead and there are COLS many of them
  for (size_t i = 0; i < (*ROWS); i++) {
    for (size_t j = 0; j < (*COLS); j++) {
      (*out)[(j * (*ROWS)) + i].re = (*in)[(i * (*COLS)) + j].re;

      if ((*conjugate)) {
        (*out)[(j * (*ROWS)) + i].im = -(*in)[(i * (*COLS)) + j].im;
      } else {
        (*out)[(j * (*ROWS)) + i].im =  (*in)[(i * (*COLS)) + j].im;
      }
    }
  }
}

void DASH_BLAS_TRANSPOSE_int_cpu(dash_cmplx_int_type** in, dash_cmplx_int_type** out, size_t* ROWS, size_t* COLS, bool* conjugate) {
  // Input is ROWS x COLS, so each "row" needs to skip COLS elements ahead and there are ROWS many of them
  // Output is COLS x ROWS, so each "row" needs to skip ROWS elements ahead and there are COLS many of them
  for (size_t i = 0; i < (*ROWS); i++) {
    for (size_t j = 0; j < (*COLS); j++) {
      (*out)[(j * (*ROWS)) + i].re = (*in)[(i * (*COLS)) + j].re;

      if ((*conjugate)) {
        (*out)[(j * (*ROWS)) + i].im = -(*in)[(i * (*COLS)) + j].im;
      } else {
        (*out)[(j * (*ROWS)) + i].im = (*in)[(i * (*COLS)) + j].im;
      }
    }
  }
}

void DASH_BLAS_MADD_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MADD_CEDR)
  DASH_BLAS_MADD_flt_cpu(A, B, C, A_ROWS, A_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_MADD", "flt", 6, A, B, C, A_ROWS, A_COLS, kernel_barrier);
#endif
}

void DASH_BLAS_MADD_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t A_ROWS, size_t A_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MADD_CEDR)
  DASH_BLAS_MADD_flt_cpu(&A, &B, &C, &A_ROWS, &A_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_MADD_flt_nb(&A, &B, &C, &A_ROWS, &A_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_BLAS_MADD_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MADD_CEDR)
  DASH_BLAS_MADD_int_cpu(A, B, C, A_ROWS, A_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_MADD", "int", 6, A, B, C, A_ROWS, A_COLS, kernel_barrier);
#endif
}

void DASH_BLAS_MADD_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t A_ROWS, size_t A_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MADD_CEDR)
  DASH_BLAS_MADD_int_cpu(&A, &B, &C, &A_ROWS, &A_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_MADD_int_nb(&A, &B, &C, &A_ROWS, &A_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_BLAS_MSUB_flt_nb(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MSUB_CEDR)
  DASH_BLAS_MSUB_flt_cpu(A, B, C, A_ROWS, A_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_MSUB", "flt", 6, A, B, C, A_ROWS, A_COLS, kernel_barrier);
#endif
}

void DASH_BLAS_MSUB_flt(dash_cmplx_flt_type* A, dash_cmplx_flt_type* B, dash_cmplx_flt_type* C, size_t A_ROWS, size_t A_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MSUB_CEDR)
  DASH_BLAS_MSUB_flt_cpu(&A, &B, &C, &A_ROWS, &A_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_MSUB_flt_nb(&A, &B, &C, &A_ROWS, &A_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_BLAS_MSUB_int_nb(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* A_ROWS, size_t* A_COLS, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MSUB_CEDR)
  DASH_BLAS_MSUB_int_cpu(A, B, C, A_ROWS, A_COLS);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_MSUB", "int", 6, A, B, C, A_ROWS, A_COLS, kernel_barrier);
#endif
}

void DASH_BLAS_MSUB_int(dash_cmplx_int_type* A, dash_cmplx_int_type* B, dash_cmplx_int_type* C, size_t A_ROWS, size_t A_COLS) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_MSUB_CEDR)
  DASH_BLAS_MSUB_int_cpu(&A, &B, &C, &A_ROWS, &A_COLS);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_MSUB_int_nb(&A, &B, &C, &A_ROWS, &A_COLS, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_BLAS_TRANSPOSE_flt_nb(dash_cmplx_flt_type** in, dash_cmplx_flt_type** out, size_t* ROWS, size_t* COLS, bool* conjugate, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_TRANSPOSE_CEDR)
  DASH_BLAS_TRANSPOSE_flt_cpu(in, out, ROWS, COLS, conjugate);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_TRANSPOSE", "flt", 6, in, out, ROWS, COLS, conjugate, kernel_barrier);
#endif
}

void DASH_BLAS_TRANSPOSE_flt(dash_cmplx_flt_type* in, dash_cmplx_flt_type* out, size_t ROWS, size_t COLS, bool conjugate) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_TRANSPOSE_CEDR)
  DASH_BLAS_TRANSPOSE_flt_cpu(&in, &out, &ROWS, &COLS, &conjugate);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_TRANSPOSE_flt_nb(&in, &out, &ROWS, &COLS, &conjugate, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_BLAS_TRANSPOSE_int_nb(dash_cmplx_int_type** in, dash_cmplx_int_type** out, size_t* ROWS, size_t* COLS, bool* conjugate, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_TRANSPOSE_CEDR)
  DASH_BLAS_TRANSPOSE_int_cpu(in, out, ROWS, COLS, conjugate);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BLAS_TRANSPOSE", "int", 6, in, out, ROWS, COLS, conjugate, kernel_barrier);
#endif
}

void DASH_BLAS_TRANSPOSE_int(dash_cmplx_int_type* in, dash_cmplx_int_type* out, size_t ROWS, size_t COLS, bool conjugate) {
#if defined(CPU_ONLY) || defined(DISABLE_BLAS_TRANSPOSE_CEDR)
  DASH_BLAS_TRANSPOSE_int_cpu(&in, &out, &ROWS, &COLS, &conjugate);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_BLAS_TRANSPOSE_int_nb(&in, &out, &ROWS, &COLS, &conjugate, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
} // Close 'extern "C"'
#endif