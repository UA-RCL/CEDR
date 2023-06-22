/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: channel_Eq.c
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

/* Include Files */
#include "channel_Eq.h"

#include "diag.h"
#include "fft.h"
#include "rdivide.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 * codegen
 * UNTITLED7 Summary of this function goes here
 *    Detailed explanation goes here
 * Arguments    : const creal_T tpilot[16]
 *                const creal_T rpilot[16]
 *                const creal_T fftout[64]
 *                const creal_T F[4096]
 *                creal_T eqData[64]
 * Return Type  : void
 */
void channel_Eq(const creal_T tpilot[16], const creal_T rpilot[16], const creal_T fftout[64], const creal_T F[4096],
                creal_T eqData[64], int mode) {
	creal_T b[256];
	creal_T dcv0[16];
	creal_T dcv1[256];
	int i0;
	int i1;
	double re;
	double im;
	int i2;
	double b_re;
	double b_im;
	double F_re;
	double F_im;
	static creal_T dcv2[64];

	/*  total number of subchannels */
	/*  Pilot Location and strength */
	/*  fft matrix   */
	/*  F = exp(2*pi*sqrt(-1)/N.* meshgrid((0:N-1),(0:N-1)).* repmat((0:N-1)',[1,N])); */

	if (mode == 0) {
		/*  Channel estimation */
		diag(tpilot, b);

		/*  estimated channel coefficient in time domain */
		/*  Equalization     */
		for (i0 = 0; i0 < 16; i0++) {
			for (i1 = 0; i1 < 16; i1++) {
				re = 0.0;
				im = 0.0;
				for (i2 = 0; i2 < 16; i2++) {
					b_re = 1.2247448713915889 * b[i1 + (i2 << 4)].re;
					b_im = 1.2247448713915889 * b[i1 + (i2 << 4)].im;
					F_re = F[i0 + ((3 + (i2 << 2)) << 6)].re;
					F_im = -F[i0 + ((3 + (i2 << 2)) << 6)].im;
					re += b_re * F_re - b_im * F_im;
					im += b_re * F_im + b_im * F_re;
				}

				dcv1[i0 + (i1 << 4)].re = re;
				dcv1[i0 + (i1 << 4)].im = -im;
			}
		}

		for (i0 = 0; i0 < 16; i0++) {
			for (i1 = 0; i1 < 16; i1++) {
				b[i1 + (i0 << 4)].re = 0.041666666666666664 * dcv1[i1 + (i0 << 4)].re;
				b[i1 + (i0 << 4)].im = 0.041666666666666664 * dcv1[i1 + (i0 << 4)].im;
			}
		}

		for (i0 = 0; i0 < 16; i0++) {
			dcv0[i0].re = 0.0;
			dcv0[i0].im = 0.0;
			for (i1 = 0; i1 < 16; i1++) {
				dcv0[i0].re += b[i0 + (i1 << 4)].re * rpilot[i1].re - b[i0 + (i1 << 4)].im * rpilot[i1].im;
				dcv0[i0].im += b[i0 + (i1 << 4)].re * rpilot[i1].im + b[i0 + (i1 << 4)].im * rpilot[i1].re;
			}
		}

		fft(dcv0, dcv2);
	} else if (mode == 1) {
		rdivide(fftout, dcv2, eqData);
	}

	/* figure; plot(abs(eqData), '-*'); grid */
}

void f2c(int indatlen, float *ibuf, creal_T *obuf) {
	int i, j;

	for (i = 0, j = 0; i < indatlen; i = i + 2) {
		obuf[j].re = ibuf[i];
		obuf[j].im = ibuf[i + 1];
		j++;
	}
}

void c2f(int indatlen, creal_T *ibuf, float *obuf) {
	int i, j;

	for (i = 0, j = 0; i < indatlen; i++) {
		obuf[j] = ibuf[i].re;
		obuf[j + 1] = ibuf[i].im;
		j = j + 2;
	}
}

/*
 * File trailer for channel_Eq.c
 *
 * [EOF]
 */
