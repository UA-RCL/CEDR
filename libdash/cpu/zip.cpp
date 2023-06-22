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

void DASH_ZIP_flt_cpu(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op) {
  dash_cmplx_flt_type temp;
  for (size_t i = 0; i < (*size); i++) {
    switch (*op) {
      case ZIP_ADD:
        (*output)[i].re = (*input_1)[i].re + (*input_2)[i].re;
        (*output)[i].im = (*input_1)[i].im + (*input_2)[i].im;
        break;
      case ZIP_SUB:
        (*output)[i].re = (*input_1)[i].re - (*input_2)[i].re;
        (*output)[i].im = (*input_1)[i].im - (*input_2)[i].im;
        break;
      case ZIP_MULT:
        temp.re = (*input_1)[i].re * (*input_2)[i].re - (*input_1)[i].im * (*input_2)[i].im;
        temp.im = (*input_1)[i].re * (*input_2)[i].im + (*input_1)[i].im * (*input_2)[i].re;
        (*output)[i].re = temp.re;
        (*output)[i].im = temp.im;
        break;
      case ZIP_DIV:
        temp.re = ( (*input_1)[i].re * (*input_2)[i].re + (*input_1)[i].im * (*input_2)[i].im)/((*input_2)[i].re*(*input_2)[i].re + (*input_2)[i].im*(*input_2)[i].im);
        temp.im = (-(*input_1)[i].re * (*input_2)[i].im + (*input_1)[i].im * (*input_2)[i].re)/((*input_2)[i].re*(*input_2)[i].re + (*input_2)[i].im*(*input_2)[i].im);
        (*output)[i].re = temp.re;
        (*output)[i].im = temp.im;
        break;
      /*case ZIP_CMP_MULT:
        (*output)[i*2] = (*input_1)[i*2] * (*input_2)[i*2] - (*input_1)[i*2+1] * (*input_2)[i*2+1];
        (*output)[i*2+1] = (*input_1)[i*2+1] * (*input_2)[i*2] + (*input_1)[i*2] * (*input_2)[i*2+1];
        break;
      */
    }
  }
}

void DASH_ZIP_int_cpu(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op) {
  dash_cmplx_int_type temp;
  for (size_t i = 0; i < (*size); i++) {
    switch (*op) {
      case ZIP_ADD:
        (*output)[i].re = (*input_1)[i].re + (*input_2)[i].re;
        (*output)[i].im = (*input_1)[i].im + (*input_2)[i].im;
        break;
      case ZIP_SUB:
        (*output)[i].re = (*input_1)[i].re - (*input_2)[i].re;
        (*output)[i].im = (*input_1)[i].im - (*input_2)[i].im;
        break;
      case ZIP_MULT:
        temp.re = (*input_1)[i].re * (*input_2)[i].re - (*input_1)[i].im * (*input_2)[i].im;
        temp.im = (*input_1)[i].re * (*input_2)[i].im + (*input_1)[i].im * (*input_2)[i].re;
        (*output)[i].re = temp.re;
        (*output)[i].im = temp.im;
        break;
      case ZIP_DIV:
        temp.re = ( (*input_1)[i].re * (*input_2)[i].re + (*input_1)[i].im * (*input_2)[i].im)/((*input_2)[i].re*(*input_2)[i].re + (*input_2)[i].im*(*input_2)[i].im);
        temp.im = (-(*input_1)[i].re * (*input_2)[i].im + (*input_1)[i].im * (*input_2)[i].re)/((*input_2)[i].re*(*input_2)[i].re + (*input_2)[i].im*(*input_2)[i].im);
        (*output)[i].re = temp.re;
        (*output)[i].im = temp.im;
        break;
    }
  }
}

void DASH_ZIP_flt_nb(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_ZIP_CEDR)
  DASH_ZIP_flt_cpu(input_1, input_2, output, size, op);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_ZIP", "flt", 6, input_1, input_2, output, size, op, kernel_barrier);
#endif
}

void DASH_ZIP_flt(dash_cmplx_flt_type* input_1, dash_cmplx_flt_type* input_2, dash_cmplx_flt_type* output, size_t size, zip_op_t op) {
#if defined(CPU_ONLY) || defined(DISABLE_ZIP_CEDR)
  DASH_ZIP_flt_cpu(&input_1, &input_2, &output, &size, &op);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_ZIP_flt_nb(&input_1, &input_2, &output, &size, &op, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_ZIP_int_nb(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_ZIP_CEDR)
  DASH_ZIP_int_cpu(input_1, input_2, output, size, op);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_ZIP", "int", 6, input_1, input_2, output, size, op, kernel_barrier);
#endif
}

void DASH_ZIP_int(dash_cmplx_int_type* input_1, dash_cmplx_int_type* input_2, dash_cmplx_int_type* output, size_t size, zip_op_t op) {
#if defined(CPU_ONLY) || defined(DISABLE_ZIP_CEDR)
  DASH_ZIP_int_cpu(&input_1, &input_2, &output, &size, &op);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_ZIP_int_nb(&input_1, &input_2, &output, &size, &op, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
} // Close 'extern "C"'
#endif
