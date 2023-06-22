#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  const size_t size = 256;

  dash_re_flt_type* input = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* output = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));

  dash_re_flt_type* input2 = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* output2 = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));

  for (int i = 0; i < 2 * size; i++) {
    input[i] = 1.0 * i;
    input2[i] = 2.0 * i;
  }

  // First, let's test two dependent API calls (FFT followed by IFFT)
  DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, true);
  DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, false);

  // Next, let's test two independent API calls
  DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, true);
  DASH_FFT_flt((dash_cmplx_flt_type*) input2, (dash_cmplx_flt_type*) output2, size, true);

  // A loop of dependent API calls
  for (int i = 0; i < 10; i++) {
    DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, true);
  }

  // A loop of independent API calls
  dash_re_flt_type* bigInput = (dash_re_flt_type*) malloc(20 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* bigOutput = (dash_re_flt_type*) malloc(20 * size * sizeof(dash_re_flt_type));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 2 * size; j++) {
      bigInput[(2 * size) * i + j] = 1.0 * j;
    }
  }
  for (int i = 0; i < 10; i++) {
    DASH_FFT_flt((dash_cmplx_flt_type*) &bigInput[(2 * size) * i], (dash_cmplx_flt_type*) &bigOutput[(2 * size) * i], size, true);
  }

  free(input);
  free(output);
  free(input2);
  free(output2);
  free(bigInput);
  free(bigOutput);
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
