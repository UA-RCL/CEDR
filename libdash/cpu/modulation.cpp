#include "dash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(CPU_ONLY)
extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);
#endif

void DASH_BPSK_flt_cpu(dash_cmplx_flt_type** input, dash_re_flt_type** output, size_t* nsymbols) {
  // These constalation values might need an update
  const dash_re_flt_type const1_r = 1.0, const1_i = 0.0;
  const dash_re_flt_type const2_r = -1.0, const2_i = 0.0;

  dash_re_flt_type dist1, dist2;
  dash_re_flt_type dist1_r, dist1_i;
  dash_re_flt_type dist2_r, dist2_i;

  for (size_t i = 0; i < (*nsymbols); i++){
    dist1_r = (*input)[i].re - const1_r;
    dist1_i = (*input)[i].im - const1_i;
    dist2_r = (*input)[i].re - const2_r;
    dist2_i = (*input)[i].im - const2_i;
    dist1 = sqrt(dist1_r*dist1_r + dist1_i*dist1_i);
    dist2 = sqrt(dist2_r*dist2_r + dist2_i*dist2_i);
    if (dist1 < dist2){
      (*output)[i] = 1;
    }
    else{
      (*output)[i] = -1;
    }
  }
}

void DASH_QAM16_flt_cpu(dash_cmplx_flt_type** input, int** output, size_t* nsymbols) {
  int CONST_SIZE = 16;

  double constReal[] = {-0.948683, -0.948683, -0.948683, -0.948683, -0.316228, -0.316228, -0.316228, -0.316228, 0.316228, 0.316228, 0.316228, 0.316228, 0.948683, 0.948683, 0.948683, 0.948683};
  double constImag[] = {-0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683};

  double* distances = (double*) calloc(CONST_SIZE, sizeof(double));

  for (size_t i = 0; i < (*nsymbols); i++) {
    for (int j = 0; j < CONST_SIZE; j++) {
      distances[j] = sqrt(pow((*input)[i].re - constReal[j], 2) + pow((*input)[i].im - constImag[j], 2));
    }

    double minVal = 2147483647;
    int minIndex = 0;
    for (int m = 0; m < CONST_SIZE; m++) {
      if (distances[m] < minVal) {
        minVal = distances[m];
        minIndex = m;
      }
    }

    (*output)[i] = minIndex;
  }

  free(distances);
}

void DASH_BPSK_flt_nb(dash_cmplx_flt_type** input, dash_re_flt_type** output, size_t* nsymbols, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_BPSK_CEDR)
  DASH_BPSK_flt_cpu(input, output, nsymbols);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_BPSK", "flt", 4, input, output, nsymbols, kernel_barrier);
#endif
}

void DASH_BPSK_flt(dash_cmplx_flt_type* input, dash_re_flt_type* output, size_t nsymbols) {
#if defined(CPU_ONLY) || defined(DISABLE_BPSK_CEDR)
  DASH_BPSK_flt_cpu(&input, &output, &nsymbols);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_BPSK_flt_nb(&input, &output, &nsymbols, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

void DASH_QAM16_flt_nb(dash_cmplx_flt_type** input, int** output, size_t* nsymbols, cedr_barrier_t* kernel_barrier) {
#if defined(CPU_ONLY) || defined(DISABLE_QAM16_CEDR)
  DASH_QAM16_flt_cpu(input, output, nsymbols);
  if (kernel_barrier != nullptr) {
    (*(kernel_barrier->completion_ctr))++;
  }
#else
  enqueue_kernel("DASH_QAM16", "flt", 4, input, output, nsymbols, kernel_barrier);
#endif
}

void DASH_QAM16_flt(dash_cmplx_flt_type* input, int* output, size_t nsymbols) {
#if defined(CPU_ONLY) || defined(DISABLE_QAM16_CEDR)
  DASH_QAM16_flt_cpu(&input, &output, &nsymbols);
#else
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  DASH_QAM16_flt_nb(&input, &output, &nsymbols, &barrier);
  
  while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
} // Close 'extern "C"'
#endif