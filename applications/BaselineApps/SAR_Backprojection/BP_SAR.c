#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../include/gsl_fft_mex.c"
#include "../include/gsl_ifft_mex.c"

#include "../include/DashExtras.h"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/SAR_Backprojection/"
#define INPUT_PHDATA PROGPATH "input_phdata.txt"
#define OUTPUT_IMAGE PROGPATH "output_img.txt"

/* Function Declarations */
void swap(double *, double *);
void fftshift(double *, double);
void linspace(double, double, double *, int);

/* Function Definitions */

void swap(double *v1, double *v2) {
	float tmp = *v1;
	*v1 = *v2;
	*v2 = tmp;
}

void fftshift(double *data, double count) {
	int k = 0;
	int c = (double)floor((float)count / 2);
	// For odd and for even numbers of element use different algorithm
	if ((int)count % 2 == 0) {
		for (k = 0; k < 2 * c; k += 2) {
			swap(&data[k], &data[k + 2 * c]);
			swap(&data[k + 1], &data[k + 1 + 2 * c]);
		}
	} else {
		double tmp1 = data[0];
		double tmp2 = data[1];
		for (k = 0; k < 2 * c; k += 2) {
			data[k] = data[2 * c + k + 2];
			data[k + 1] = data[2 * c + k + 3];
			data[2 * c + k + 2] = data[k + 2];
			data[2 * c + k + 3] = data[k + 3];
		}
		data[2 * c] = tmp1;
		data[2 * c] = tmp2;
	}
}

void linspace(double x, double y, double *q, int l) {
	double d;
	d = (double)(y - x) / (double)(l - 1);
	for (int i = 0; i < l; i++) {
		q[i] = x + i * d;
	}
}

int main(int argc, char *argv[]) {
	double c = 3e8;
	double PRI = atof(argv[1]);  // Pulse Repetition Interval
	double z = atof(argv[2]);    // Height of platform
	double v = atof(argv[3]);    // Velocity of platform
	int Np = atoi(argv[4]);      // Number of pulses
	int N_fft = atoi(argv[5]);   // FFT size
	double Xmin = atof(argv[6]);
	double Xmax = atof(argv[7]);  // Target area in azimuth direction
	double Yc = atof(argv[8]);
	double Y0 = atof(argv[9]);   // Target in [Yc-Y0, Yc+Y0]
	int s = atoi(argv[10]);      // Target area size
	double fc = atof(argv[11]);  // Carrier frequency
	double B = atof(argv[12]);   // Bandwidth
	double *x = malloc(Np * sizeof(double));
	x[0] = 0;
	for (int i = 1; i < Np; i++) {
		x[i] = x[i - 1] + v * PRI;  // Platform position in azimuth direction for every pulse
	}
	double Wr;
	Wr = c / (2 * B / (N_fft - 1));
	double *r_vec = malloc(N_fft * sizeof(double));  // range wrath
	linspace(-Wr / 2, Wr * (N_fft / 2 - 1) / N_fft, r_vec, N_fft);
	double *x_mat = malloc(s * sizeof(double));  // target area indices in azimuth direction
	double *y_mat = malloc(s * sizeof(double));  // target area indices in range direction
	x_mat[0] = Xmin;
	y_mat[0] = Yc - Y0;
	for (int i = 1; i < s; i++) {
		x_mat[i] = x_mat[i - 1] + (Xmax - Xmin) / (s - 1);
		y_mat[i] = y_mat[i - 1] + 2 * Y0 / (s - 1);
	}
	double *im_final = malloc(s * s * sizeof(double));       // image output
	double *im_final2 = malloc(2 * s * s * sizeof(double));  // image output in complex form
	for (int i = 0; i < 2 * s * s; i++) {
		im_final2[i] = 0;
	}
	double *ifft_arr = malloc(2 * N_fft * sizeof(double));  // IFFT input
	double *rc = malloc(2 * N_fft * sizeof(double));        // IFFT output
	double *rc_re = malloc(N_fft * sizeof(double));         // real part of IFFT
	double *rc_im = malloc(N_fft * sizeof(double));         // imaginary part of IFFT
	double *dR = malloc(s * s * sizeof(double));            // Differential range
	double *phCorr = malloc(2 * s * s * sizeof(double));    // corrected phase
	double v1, v2;
	double *phdata = malloc(2 * N_fft * Np * sizeof(double));  // phase history data
	FILE *fp;
	fp = fopen(INPUT_PHDATA, "r");
	for (int i = 0; i < 2 * N_fft * Np; i++) {
		fscanf(fp, "%lf", &phdata[i]);
	}
	fclose(fp);
	printf("test");
	for (int i = 0; i < Np; i++) {
		for (int j = 0; j < 2 * N_fft; j += 2) {
			ifft_arr[j] = phdata[Np * j + 2 * i];
			ifft_arr[j + 1] = phdata[Np * j + 2 * i + 1];
		}
        KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", N_fft));
		gsl_ifft(ifft_arr, rc, N_fft);
        KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", N_fft));
		fftshift(rc, N_fft);
		for (int j2 = 0; j2 < 2 * N_fft; j2 += 2) {
			rc_re[j2 / 2] = rc[j2];
			rc_im[j2 / 2] = rc[j2 + 1];
		}
		for (int k1 = 0; k1 < s; k1++) {
			for (int k2 = 0; k2 < s; k2++) {
				dR[k1 + k2 * s] = sqrt((x_mat[k2] - x[i]) * (x_mat[k2] - x[i]) + y_mat[k1] * y_mat[k1] + z * z) -
				                  sqrt(x[i] * x[i] + z * z);
			}
		}
		for (int k3 = 0; k3 < 2 * s * s; k3 += 2) {
			phCorr[k3] = cos(4 * M_PI * (fc - B / 2) / c * dR[k3 / 2]);
			phCorr[k3 + 1] = sin(4 * M_PI * (fc - B / 2) / c * dR[k3 / 2]);
		}
		// linear interpolation
		for (int k4 = 0; k4 < s * s; k4++) {
			if (dR[k4] > r_vec[0] && dR[k4] < r_vec[N_fft - 1]) {
				int i3 = 1;
				while (dR[k4] > r_vec[i3]) {
					i3 = i3 + 1;
				}
				v1 = rc_re[i3 - 1] +
				     (rc_re[i3] - rc_re[i3 - 1]) / (r_vec[i3] - r_vec[i3 - 1]) * (dR[k4] - r_vec[i3 - 1]);
				v2 = rc_im[i3 - 1] +
				     (rc_im[i3] - rc_im[i3 - 1]) / (r_vec[i3] - r_vec[i3 - 1]) * (dR[k4] - r_vec[i3 - 1]);
				im_final2[2 * k4] = im_final2[2 * k4] + v1 * phCorr[2 * k4] - v2 * phCorr[2 * k4 + 1];
				im_final2[2 * k4 + 1] = im_final2[2 * k4 + 1] + v1 * phCorr[2 * k4 + 1] + v2 * phCorr[2 * k4];
			}
		}
	}
	for (int i2 = 0; i2 < s * s; i2++) {
		im_final[i2] = sqrt(im_final2[2 * i2] * im_final2[2 * i2] + im_final2[2 * i2 + 1] * im_final2[2 * i2 + 1]);
	}
	fp = fopen(OUTPUT_IMAGE, "w");
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			fprintf(fp, "%lf ", im_final[i + j * s]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}
