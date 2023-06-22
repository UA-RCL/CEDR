/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: channel_Eq.h
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

#ifndef __CHANNEL_EQ_H__
#define __CHANNEL_EQ_H__

/* Include Files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "channel_Eq_types.h"
#include "rtwtypes.h"

/* Function Declarations */
extern void channel_Eq(const creal_T tpilot[16], const creal_T rpilot[16], const creal_T fftout[64],
                       const creal_T F[4096], creal_T eqData[64], int mode);

void f2c(int indatlen, float *ibuf, creal_T *obuf);

void c2f(int indatlen, creal_T *ibuf, float *obuf);

#endif

/*
 * File trailer for channel_Eq.h
 *
 * [EOF]
 */
