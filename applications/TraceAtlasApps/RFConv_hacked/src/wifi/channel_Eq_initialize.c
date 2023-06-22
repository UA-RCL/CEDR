/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: channel_Eq_initialize.c
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

/* Include Files */
#include "channel_Eq_initialize.h"

#include "channel_Eq.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 * Arguments    : void
 * Return Type  : void
 */
void channel_Eq_initialize(void) { rt_InitInfAndNaN(8U); }

/*
 * File trailer for channel_Eq_initialize.c
 *
 * [EOF]
 */
