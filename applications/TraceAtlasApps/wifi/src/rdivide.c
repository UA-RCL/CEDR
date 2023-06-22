/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: rdivide.c
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "channel_Eq.h"
#include "rdivide.h"

/* Function Definitions */

/*
 * Arguments    : const creal_T x[64]
 *                const creal_T y[64]
 *                creal_T z[64]
 * Return Type  : void
 */
void rdivide(const creal_T x[64], const creal_T y[64], creal_T z[64])
{
  int i;
  double brm;
  double bim;
  double d;
  for (i = 0; i < 64; i++) {
    if (y[i].im == 0.0) {
      if (x[i].im == 0.0) {
        z[i].re = x[i].re / y[i].re;
        z[i].im = 0.0;
      } else if (x[i].re == 0.0) {
        z[i].re = 0.0;
        z[i].im = x[i].im / y[i].re;
      } else {
        z[i].re = x[i].re / y[i].re;
        z[i].im = x[i].im / y[i].re;
      }
    } else if (y[i].re == 0.0) {
      if (x[i].re == 0.0) {
        z[i].re = x[i].im / y[i].im;
        z[i].im = 0.0;
      } else if (x[i].im == 0.0) {
        z[i].re = 0.0;
        z[i].im = -(x[i].re / y[i].im);
      } else {
        z[i].re = x[i].im / y[i].im;
        z[i].im = -(x[i].re / y[i].im);
      }
    } else {
      brm = fabs(y[i].re);
      bim = fabs(y[i].im);
      if (brm > bim) {
        bim = y[i].im / y[i].re;
        d = y[i].re + bim * y[i].im;
        z[i].re = (x[i].re + bim * x[i].im) / d;
        z[i].im = (x[i].im - bim * x[i].re) / d;
      } else if (bim == brm) {
        if (y[i].re > 0.0) {
          bim = 0.5;
        } else {
          bim = -0.5;
        }

        if (y[i].im > 0.0) {
          d = 0.5;
        } else {
          d = -0.5;
        }

        z[i].re = (x[i].re * bim + x[i].im * d) / brm;
        z[i].im = (x[i].im * bim - x[i].re * d) / brm;
      } else {
        bim = y[i].re / y[i].im;
        d = y[i].im + bim * y[i].re;
        z[i].re = (bim * x[i].re + x[i].im) / d;
        z[i].im = (bim * x[i].im - x[i].re) / d;
      }
    }
  }
}

/*
 * File trailer for rdivide.c
 *
 * [EOF]
 */
