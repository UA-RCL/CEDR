#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/DashExtras.h"
#include "../include/gsl_fft_mex.c"
#include "../include/gsl_ifft_mex.c"
#include "lfm_gen.c"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/radar_correlator/"
#define TIMEIN PROGPATH "time_input.txt"
#define RXIN PROGPATH "received_input.txt"
#define LAGOUT PROGPATH "lag_output.txt"

/*const float complex I;*/

void xcorr(double *, double *, size_t, double *);

double radar_Rx(double *, double *, double, double, double, double *, size_t, size_t);

void xcorr(double *x, double *y, size_t n_samp, double *corr) {
	size_t len = 2 * n_samp - 1;

	double *c = malloc(2 * len * sizeof(double));
	double *d = malloc(2 * len * sizeof(double));

	size_t x_count = 0;
	size_t y_count = 0;

	for (size_t i = 0; i < 2 * len; i += 2) {
		if (i / 2 > n_samp - 1) {
			c[i] = x[x_count];
			c[i + 1] = x[x_count + 1];
			x_count += 2;
		} else {
			c[i] = 0;
			c[i + 1] = 0;
		}

		if (i > n_samp) {
			d[i] = 0;
			d[i + 1] = 0;
		} else {
			d[i] = y[y_count];
			d[i + 1] = y[y_count + 1];
			y_count += 2;
		}
	}

	double *X1 = malloc(2 * len * sizeof(double));
	double *X2 = malloc(2 * len * sizeof(double));
	double *corr_freq = malloc(2 * len * sizeof(double));
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]",len));
	gsl_fft(c, X1, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]",len));
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]",len));
	gsl_fft(d, X2, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]",len));

    KERN_ENTER(make_label("ZIP[multiply][complex][float64][%d]",len));
	for (size_t i = 0; i < 2 * len; i += 2) {
		corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
		corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
	}
    KERN_EXIT(make_label("ZIP[multiply][complex][float64][%d]",len));

    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]",len));
	gsl_ifft(corr_freq, corr, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]",len));
}

double radar_Rx(double *received_signal, double *time, double B, double T, double samp_rate, double *corr,
                size_t n_samp, size_t time_n_samp) {
	double *gen_wave = malloc(2 * n_samp * sizeof(double));
	waveform_gen(time, B, T, gen_wave, time_n_samp);

	for (size_t i = time_n_samp; i < n_samp; i++) {
		gen_wave[i] = 0;
	}

	double lag;
	// Add code for zero-padding, to make sure signals are of same length
	xcorr(received_signal, gen_wave, n_samp, corr);

	// Code to find maximum
	double max_corr = 0,tmp=0;
	double index = 0;
	for (size_t i = 0; i < 2 * (2 * n_samp - 1); i += 2) {
		// Only finding maximum of real part of correlation
		tmp = corr[i]*corr[i] + corr[i+1]*corr[i+1];
		if (corr[i] > max_corr) {
			max_corr = corr[i];
			index = i / 2;
		}
	}
	
	lag = (index - n_samp) / samp_rate;
	return lag;
}

int main(int argc, char *argv[]) {
	// MAIN FUNCTION TO ACCEPT INPUTS FROM FILE

	// order of arguments: number of samples,B,T,sampling_rate
	size_t n_samples = atoi(argv[1]);
	size_t time_n_samples = atoi(argv[2]);
	double T = atof(argv[3]);
	double B = atof(argv[4]);
	double sampling_rate = atof(argv[5]);

	double *time = malloc(n_samples * sizeof(double));
	;
	double *received = malloc(2 * n_samples * sizeof(double));

	FILE *fp;
	fp = fopen(TIMEIN, "r");

	for (size_t i = 0; i < n_samples; i++) {
		fscanf(fp, "%lf", &time[i]);
	}
	fclose(fp);

	fp = fopen(RXIN, "r");

	for (size_t i = 0; i < 2 * n_samples; i++) {
		fscanf(fp, "%lf", &received[i]);
	}
	fclose(fp);

	double lag;
	double *corr = malloc((2 * (2 * n_samples - 1)) * sizeof(double));

	lag = radar_Rx(received, time, B, T, sampling_rate, corr, n_samples, time_n_samples);

	fp = fopen(LAGOUT, "w");

	fprintf(fp, "Lag Value is: %lf", lag);
	fclose(fp);
}
