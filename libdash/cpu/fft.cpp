#include "dash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gsl/gsl_fft_complex_float.h>
#include <gsl/gsl_fft_complex.h>
#include <pthread.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(CPU_ONLY)
extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);
#endif

void DASH_FFT_flt_cpu(dash_cmplx_flt_type** input, dash_cmplx_flt_type** output, size_t* size, bool* isForwardTransform) {
//  printf("[fft] Running a %lu-Pt %s on the CPU\n", (*size), *isForwardTransform ? "FFT" : "IFFT");
  
  // Note: we copy to a new buffer here because the API doesn't state that we modify the input
  dash_cmplx_flt_type *data = (dash_cmplx_flt_type*) malloc((*size) * sizeof(dash_cmplx_flt_type));
  memcpy(data, (*input), (*size) * sizeof(dash_cmplx_flt_type));
  
  int check;
  // Our base floating point type is 4-byte (float)
  if (sizeof(dash_re_flt_type) == 4) {
    if (*isForwardTransform) {
      check = gsl_fft_complex_float_radix2_forward((float*)(data), 1, (*size));
    } else {
      check = gsl_fft_complex_float_radix2_inverse((float*)(data), 1, (*size));
    }
  } 
  // Otherwise it's 8-byte (double)
  else {
    if (*isForwardTransform) {
      check = gsl_fft_complex_radix2_forward((double*)(data), 1, (*size));
    } else {
      check = gsl_fft_complex_radix2_inverse((double*)(data), 1, (*size));
    }
  }

  if (check != 0) {
    fprintf(stderr, "[libdash] Failed to complete DASH_FFT_flt_cpu using libgsl with message %d!\n", check);
    free(data);
    return;
  }

  memcpy((*output), data, (*size) * sizeof(dash_cmplx_flt_type));
  free(data);
  return;
}

void DASH_FFT_int_cpu(dash_cmplx_int_type** input, dash_cmplx_int_type** output, size_t* size, bool* isForwardTransform) {
  //printf("[fft] Running a %d-Pt %s on the CPU\n", (int)(*size), *isForwardTransform ? "FFT" : "IFFT");

  dash_cmplx_flt_type *data = (dash_cmplx_flt_type*) malloc((*size) * sizeof(dash_cmplx_flt_type));
  for (size_t i = 0; i < (*size); i++) {
    data[i].re = (dash_re_flt_type) (*input)[i].re;
    data[i].im = (dash_re_flt_type) (*input)[i].im;
  }

  int check;
  // Our base floating point type is 4-byte (float)
  if (sizeof(dash_re_flt_type) == 4) {
    if (*isForwardTransform) {
      check = gsl_fft_complex_float_radix2_forward((float*)(data), 1, (*size));
    } else {
      check = gsl_fft_complex_float_radix2_inverse((float*)(data), 1, (*size));
    }
  } 
  // Otherwise it's 8-byte (double)
  else {
    if (*isForwardTransform) {
      check = gsl_fft_complex_radix2_forward((double*)(data), 1, (*size));
    } else {
      check = gsl_fft_complex_radix2_inverse((double*)(data), 1, (*size));
    }
  }

  if (check != 0){
    fprintf(stderr, "[libdash] Failed to complete DASH_FFT_int_cpu using libgsl with message %d!\n", check);
    free(data);
    return;
  }

  for (size_t i=0; i < (*size); i++) {
    (*output)[i].re = (dash_re_int_type) data[i].re;  
    (*output)[i].im = (dash_re_int_type) data[i].im;
  }
  free(data);
  return;
}

void DASH_FFT_flt_nb(dash_cmplx_flt_type** input, dash_cmplx_flt_type** output, size_t* size, bool* isForwardTransform, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_FFT_CEDR)
  DASH_FFT_flt_cpu(input, output, size, isForwardTransform);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_FFT", "flt", 5, input, output, size, isForwardTransform, kernel_barrier);
#endif
}

void DASH_FFT_flt(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool isForwardTransform) {
#if defined(CPU_ONLY) || defined(DISABLE_FFT_CEDR)
  DASH_FFT_flt_cpu(&input, &output, &size, &isForwardTransform);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_FFT_flt_nb(&input, &output, &size, &isForwardTransform, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_FFT_int_nb(dash_cmplx_int_type** input, dash_cmplx_int_type** output, size_t* size, bool* isForwardTransform, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_FFT_CEDR)
  DASH_FFT_int_cpu(input, output, size, isForwardTransform);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_FFT", "int", 5, input, output, size, isForwardTransform, kernel_barrier);
#endif
}

void DASH_FFT_int(dash_cmplx_int_type* input, dash_cmplx_int_type* output, size_t size, bool isForwardTransform) {
#if defined(CPU_ONLY) || defined(DISABLE_FFT_CEDR)
  DASH_FFT_int_cpu(&input, &output, &size, &isForwardTransform);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);
  
  DASH_FFT_int_nb(&input, &output, &size, &isForwardTransform, &barrier);

  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
} // Close 'extern "C"'
#endif
