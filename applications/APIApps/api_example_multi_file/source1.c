#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "dash.h"

void function_in_other_file(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool forwardTrans);

int main(void) {
printf("Starting execution of the non-kernel thread [nk]\n");

  printf("[nk] I'm preparing data\n");

  dash_re_flt_type* input = (dash_re_flt_type*) malloc(64 * sizeof(dash_re_flt_type));
  dash_re_flt_type* output = (dash_re_flt_type*) malloc(64 * sizeof(dash_re_flt_type));
  size_t size = 32;
  bool forwardTrans = true;

  for (int i = 0; i < 64; i++) {
    input[i] = 1.0 * i;
  }
  printf("[nk] Input preparation complete\n");

  printf("[nk] Launching my kernel that was replaced from i.e. DASH_FFT\n");

  for (int i = 0; i < 10; i++) {
    DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);
  }

  function_in_other_file((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);

  printf("[nk] Kernel execution is complete! Printing output...\n");

  for (int i = 0; i < 64; i++) {
    printf("%lf ", output[i]);
  }
  printf("\n");

  free(input);
  free(output);
  printf("[nk] Non-kernel thread execution is complete...\n\n");

  return 0;
}
