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

void DASH_CONV_2D_int_cpu(dash_re_int_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_int_type **output) {
  int i, j, k, l;
  int s, w;
  int z;

  float sum;

  z = (*mask_size) / 2;

  for (i = 0; i < (*height); i++) {
    for (j = 0; j < (*width); j++) {
      sum = 0.0;
      for (k = 0; k < (*mask_size); k++) {
        for (l = 0; l < (*mask_size); l++) {
          s = i + k - z;
          w = j + l - z;
          if ((s >= 0 && s < (*height)) && (w >= 0 && w < (*width))) {
            sum += (*input)[(*width) * s + w] * (*mask)[(*mask_size) * k + l];
          }
        }
      }
      (*output)[i * (*width) + j] = ( (sum > CONV_2D_MAX) ? CONV_2D_MAX : ( (sum < CONV_2D_MIN) ? CONV_2D_MAX : sum));
    }
  }
}

void DASH_CONV_1D_flt_cpu(dash_re_flt_type** arr, int* size, dash_re_flt_type** mask, int* mask_size, dash_re_flt_type** result){
   int n = (*mask_size)/2;
    int i,j;
     for (i=0;i<(*size);i++)
     {
        for(j=0;j<(*mask_size);j++)
        {
            if(i-n+j <0 || i-n+j >(*size))
                {
                  continue;
                }
            else
                  {
                (*result)[i] += (*arr)[i-n+j] * (*mask)[j];
                  }
        }
     }
}

void DASH_CONV_2D_int_nb(dash_re_int_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_int_type **output, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_CONV_2D_CEDR)
  DASH_CONV_2D_int_cpu(input, height, width, mask, mask_size, output);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_CONV_2D", "int", 7, input, height, width, mask, mask_size, output, kernel_barrier);
#endif
}

void DASH_CONV_2D_int(dash_re_int_type *input, int height, int width, dash_re_flt_type *mask, int mask_size, dash_re_int_type *output) {
#if defined(CPU_ONLY) || defined(DISABLE_CONV_2D_CEDR)
  DASH_CONV_2D_int_cpu(&input, &height, &width, &mask, &mask_size, &output);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_CONV_2D_int_nb(&input, &height, &width, &mask, &mask_size, &output, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_CONV_1D_flt_nb(dash_re_flt_type** input, int* size, dash_re_flt_type** mask, int* mask_size, dash_re_flt_type** output, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_CONV_1D_CEDR)
  DASH_CONV_1D_flt_cpu(input, size, mask, mask_size, output);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_CONV_1D", "flt", 6, input, size, mask, mask_size, output, kernel_barrier);
#endif
}

void DASH_CONV_1D_flt(dash_re_flt_type* input, int size, dash_re_flt_type* mask, int mask_size, dash_re_flt_type* output) {
#if defined(CPU_ONLY) || defined(DISABLE_CONV_1D_CEDR)
  DASH_CONV_1D_flt_cpu(&input, &size, &mask, &mask_size, &output);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_CONV_1D_flt_nb(&input, &size, &mask, &mask_size, &output, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}
   
#if defined(__cplusplus)
} // Close 'extern "C"'
#endif
