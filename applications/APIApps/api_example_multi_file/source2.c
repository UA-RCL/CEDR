#include "dash.h"

void function_in_other_file(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool isForwardTrans) {
  DASH_FFT_flt(input, output, size, isForwardTrans);
}
