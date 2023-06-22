/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fft.c
 *
 * MATLAB Coder version            : 3.0
 * C/C++ source code generated on  : 24-Oct-2015 18:20:13
 */

/* Include Files */
#include "fft.h"

#include "channel_Eq.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 * Arguments    : const creal_T x[16]
 *                creal_T y[64]
 * Return Type  : void
 */
void fft(const creal_T x[16], creal_T y[64]) {
	int i;
	int ix;
	int ju;
	int iy;
	boolean_T tst;
	double temp_re;
	double temp_im;
	int iheight;
	int istart;
	int j;
	static const double dv0[33] = {1.0,
	                               0.99518472667219693,
	                               0.98078528040323043,
	                               0.95694033573220882,
	                               0.92387953251128674,
	                               0.881921264348355,
	                               0.83146961230254524,
	                               0.773010453362737,
	                               0.70710678118654757,
	                               0.63439328416364549,
	                               0.55557023301960218,
	                               0.47139673682599764,
	                               0.38268343236508978,
	                               0.29028467725446233,
	                               0.19509032201612825,
	                               0.0980171403295606,
	                               0.0,
	                               -0.0980171403295606,
	                               -0.19509032201612825,
	                               -0.29028467725446233,
	                               -0.38268343236508978,
	                               -0.47139673682599764,
	                               -0.55557023301960218,
	                               -0.63439328416364549,
	                               -0.70710678118654757,
	                               -0.773010453362737,
	                               -0.83146961230254524,
	                               -0.881921264348355,
	                               -0.92387953251128674,
	                               -0.95694033573220882,
	                               -0.98078528040323043,
	                               -0.99518472667219693,
	                               -1.0};

	double twid_re;
	static const double dv1[33] = {0.0,
	                               -0.0980171403295606,
	                               -0.19509032201612825,
	                               -0.29028467725446233,
	                               -0.38268343236508978,
	                               -0.47139673682599764,
	                               -0.55557023301960218,
	                               -0.63439328416364549,
	                               -0.70710678118654757,
	                               -0.773010453362737,
	                               -0.83146961230254524,
	                               -0.881921264348355,
	                               -0.92387953251128674,
	                               -0.95694033573220882,
	                               -0.98078528040323043,
	                               -0.99518472667219693,
	                               -1.0,
	                               -0.99518472667219693,
	                               -0.98078528040323043,
	                               -0.95694033573220882,
	                               -0.92387953251128674,
	                               -0.881921264348355,
	                               -0.83146961230254524,
	                               -0.773010453362737,
	                               -0.70710678118654757,
	                               -0.63439328416364549,
	                               -0.55557023301960218,
	                               -0.47139673682599764,
	                               -0.38268343236508978,
	                               -0.29028467725446233,
	                               -0.19509032201612825,
	                               -0.0980171403295606,
	                               -0.0};

	double twid_im;
	int ihi;
	for (i = 0; i < 64; i++) {
		y[i].re = 0.0;
		y[i].im = 0.0;
	}

	ix = 0;
	ju = 0;
	iy = 0;
	for (i = 0; i < 15; i++) {
		y[iy] = x[ix];
		iy = 64;
		tst = true;
		while (tst) {
			iy >>= 1;
			ju ^= iy;
			tst = ((ju & iy) == 0);
		}

		iy = ju;
		ix++;
	}

	y[iy] = x[ix];
	for (i = 0; i <= 63; i += 2) {
		temp_re = y[i + 1].re;
		temp_im = y[i + 1].im;
		y[i + 1].re = y[i].re - y[i + 1].re;
		y[i + 1].im = y[i].im - y[i + 1].im;
		y[i].re += temp_re;
		y[i].im += temp_im;
	}

	iy = 2;
	ix = 4;
	ju = 16;
	iheight = 61;
	while (ju > 0) {
		for (i = 0; i < iheight; i += ix) {
			temp_re = y[i + iy].re;
			temp_im = y[i + iy].im;
			y[i + iy].re = y[i].re - temp_re;
			y[i + iy].im = y[i].im - temp_im;
			y[i].re += temp_re;
			y[i].im += temp_im;
		}

		istart = 1;
		for (j = ju; j < 32; j += ju) {
			twid_re = dv0[j];
			twid_im = dv1[j];
			i = istart;
			ihi = istart + iheight;
			while (i < ihi) {
				temp_re = twid_re * y[i + iy].re - twid_im * y[i + iy].im;
				temp_im = twid_re * y[i + iy].im + twid_im * y[i + iy].re;
				y[i + iy].re = y[i].re - temp_re;
				y[i + iy].im = y[i].im - temp_im;
				y[i].re += temp_re;
				y[i].im += temp_im;
				i += ix;
			}

			istart++;
		}

		ju /= 2;
		iy = ix;
		ix <<= 1;
		iheight -= iy;
	}
}

/*
 * File trailer for fft.c
 *
 * [EOF]
 */
