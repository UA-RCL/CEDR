/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: diag.c
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "channel_Eq.h"
#include "diag.h"

/* Function Definitions */

/*
 * Arguments    : const creal_T v[16]
 *                creal_T d[256]
 * Return Type  : void
 */
void diag(const creal_T v[16], creal_T d[256])
{
  int j;
  for (j = 0; j < 256; j++) {
    d[j].re = 0.0;
    d[j].im = 0.0;
  }

  for (j = 0; j < 16; j++) {
    d[j + (j << 4)] = v[j];
  }
}

/*
 * File trailer for diag.c
 *
 * [EOF]
 */
