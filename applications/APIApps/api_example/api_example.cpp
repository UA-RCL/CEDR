#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  printf("[nk] I'm preparing data\n");

  const size_t size = 256;
  dash_re_flt_type* input = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* output = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  bool forwardTrans = true;

  for (int i = 0; i < 2 * size; i++) {
    input[i] = 1.0 * i;
  }
  printf("[nk] Input preparation complete\n");

  printf("[nk] Launching my kernel that was replaced from i.e. DASH_FFT\n");

  for (int i = 0; i < 10; i++) {
    printf("Launching FFT number %d\n", i);
    DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);
    sleep(1);
  }

  printf("[nk] Kernel execution is complete! Printing output...\n");

  for (int i = 0; i < 2 * size; i++) {
    printf("%lf ", output[i]);
  }
  printf("\n");

  printf("\n\n");
  printf("[nk] Launching mmult...\n");
  
  // 4x64 * 64x4 matrices
  dash_cmplx_flt_type* A = (dash_cmplx_flt_type*) calloc(4 * 64, sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type* B = (dash_cmplx_flt_type*) calloc(4 * 64, sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type* C = (dash_cmplx_flt_type*) calloc(4 * 4, sizeof(dash_cmplx_flt_type));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 64; j++) {
      A[j + 64 * i].re = (j + 64 * i);
      A[j + 64 * i].im = (j + 64 * i);
    }
  }
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 4; j++) {
      B[j + 4 * i].re = (j + 4 * i);
      B[j + 4 * i].im = (j + 4 * i);
    }
  }
  printf("Computing GEMM...\n");
  DASH_GEMM_flt(A, B, C, 4, 64, 4);
  printf("Printing results...\n");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%f ", C[j + 4 * i].re);
    }
    printf("\n");
  }
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%f ", C[j + 4 * i].im);
    }
    printf("\n");
  }
  free(A);
  free(B);
  free(C);

  free(input);
  free(output);
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
