#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"

size_t FFT_SIZE = 32;
size_t NUM_FFTS = 8;
bool   is_fwd   = true;

int main(void) {
  dash_cmplx_flt_type **fft_inputs  = (dash_cmplx_flt_type**) calloc(NUM_FFTS, sizeof(dash_cmplx_flt_type*));
  dash_cmplx_flt_type **fft_outputs = (dash_cmplx_flt_type**) calloc(NUM_FFTS, sizeof(dash_cmplx_flt_type*));

  printf("Initializing input for %ld FFTs...\n", NUM_FFTS);

  for (size_t i = 0; i < NUM_FFTS; i++) {
    fft_inputs[i]  = (dash_cmplx_flt_type*) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
    fft_outputs[i] = (dash_cmplx_flt_type*) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  }

  for (size_t i = 0; i < NUM_FFTS; i++) {
    for (size_t j = 0; j < FFT_SIZE; j++) {
      if (j == 0) {
        fft_inputs[i][j] = {.re = 1.0f, .im = 0.0f};
      } else {
        fft_inputs[i][j] = {.re = 0.0f, .im = 0.0f};
      }
      //printf("The address of element (%ld, %ld) is: %p\n", i, j, &fft_inputs[i][j]);
    }
  }
  //printf("The address of the 7th row in this array is: %p\n", &(fft_inputs[7]));

  printf("Initializing barrier logic, calling non-blocking APIs, and awaiting completion...\n");

  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (size_t i = 0; i < NUM_FFTS; i++) {
    printf("calling the %ld-th API\n", i);
    DASH_FFT_flt_nb(&fft_inputs[i], &fft_outputs[i], &FFT_SIZE, &is_fwd, &barrier);
  }

  while (completion_ctr != NUM_FFTS) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
    printf("%u FFTs have been completed...\n", completion_ctr);
  }
  pthread_mutex_unlock(barrier.mutex);

  printf("All %ld FFTs have been completed! Printing results...\n", NUM_FFTS);

  for (size_t i = 0; i < NUM_FFTS; i++) {
    printf("FFT %ld: ", i);
    for (size_t j = 0; j < FFT_SIZE; j++) {
      printf("(%f, %f)", fft_outputs[i][j].re, fft_outputs[i][j].im);
      if (j != FFT_SIZE - 1) {
        printf(", ");
      }
    }
    printf("\n");
  }

  for (size_t i = 0; i < NUM_FFTS; i++) {
    free(fft_inputs[i]);
    free(fft_outputs[i]);
  }
  free(fft_inputs);
  free(fft_outputs);

  return 0;
}