#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/DashExtras.h"
#include "../include/gsl_fft_mex.c"
#include "../include/gsl_ifft_mex.c"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/pulse_doppler/"
#define PDPULSE PROGPATH "input_pd_pulse.txt"
#define PDPS PROGPATH "input_pd_ps.txt"
#define OUTPUT PROGPATH "output_pd_f.txt"

/* Function Declarations */
void xcorr(double *, double *, size_t, double *);
void swap(double *, double *);
void fftshift(double *, double);

/* Function Definitions */
void xcorr(double *x, double *y, size_t n_samp, double *corr) {
	size_t len;
	len = 2 * n_samp;
	double *c = malloc(2 * len * sizeof(double));
	double *d = malloc(2 * len * sizeof(double));

	size_t x_count = 0;
	size_t y_count = 0;

	double *z = malloc(2 * (n_samp) * sizeof(double));
	for (size_t i = 0; i < 2 * (n_samp); i += 2) {
		z[i] = 0;
		z[i + 1] = 0;
	}
	for (size_t i = 0; i < 2 * (n_samp - 1); i += 2) {
		c[i] = 0;
		c[i + 1] = 0;
	}
	memcpy(c + 2 * (n_samp - 1), x, 2 * n_samp * sizeof(double));
	c[2 * len - 2] = 0;
	c[2 * len - 1] = 0;
	memcpy(d, y, 2 * n_samp * sizeof(double));
	memcpy(d + 2 * n_samp, z, 2 * (n_samp) * sizeof(double));
	double *X1 = malloc(2 * len * sizeof(double));
	double *X2 = malloc(2 * len * sizeof(double));
	double *corr_freq = malloc(2 * len * sizeof(double));
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]",2*n_samp));
	gsl_fft(c, X1, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]",2*n_samp));
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]",2*n_samp));
	gsl_fft(d, X2, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]",2*n_samp));
	free(c);
	free(d);
	free(z);
    KERN_ENTER(make_label("ZIP[multiply][complex][float64][%d]",len));
	for (size_t i = 0; i < 2 * len; i += 2) {
		corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
		corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
	}
    KERN_EXIT(make_label("ZIP[multiply][complex][float64][%d]",len));
	free(X1);
	free(X2);
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]",2*n_samp));
	gsl_ifft(corr_freq, corr, len);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]",2*n_samp));

	free(corr_freq);
}

void swap(double *v1, double *v2) {
	double tmp = *v1;
	*v1 = *v2;
	*v2 = tmp;
}

void fftshift(double *data, double count) {
	int k = 0;
	int c = (double)floor((float)count / 2);
	// For odd and for even numbers of element use different algorithm
	if ((int)count % 2 == 0) {
		for (k = 0; k < c; k++) swap(&data[k], &data[k + c]);
	} else {
		double tmp = data[0];
		for (k = 0; k < c; k++) {
			data[k] = data[c + k + 1];
			data[c + k + 1] = data[k + 1];
		}
		data[c] = tmp;
	}
}

int main(int argc, char *argv[]) {
	size_t m = atoi(argv[1]);          // number of pulses
	size_t n_samples = atoi(argv[2]);  // length of single pulse
	double PRI = atof(argv[3]);
	int i, j, k, n, x, y, z, o;

	double *mf = malloc((2 * n_samples) * m * 2 * sizeof(double));  // build a 2D array for the output of the matched
	                                                                // filter
	double *p = malloc(2 * n_samples * sizeof(double));             // array for pulse with noise
	double *pulse = malloc(2 * n_samples * sizeof(double));         // array for the original pulse
	double *corr = malloc(2 * (2 * n_samples) * sizeof(double));    // array for the output of matched filter

	// creat plans for FFT in matched filter
	
    FILE *fp;
	fp = fopen(PDPULSE, "r");  // read the original pulse
	for (i = 0; i < 2 * n_samples; i++) {
		fscanf(fp, "%lf", &pulse[i]);
	}
	fclose(fp);

	/* matched filter */

	fp = fopen(PDPS, "r");  // read the multiple pulses with noise and delay
	for (k = 0; k < m; k++) {
		for (j = 0; j < 2 * n_samples; j++) {
			fscanf(fp, "%lf", &p[j]);
		}

		/* matched filter */

		xcorr(p, pulse, n_samples, corr);

		/* put the output into a new 2D array */

		for (n = 0; n < 2 * (2 * n_samples); n += 2) {
			mf[n / 2 + (2 * k) * (2 * n_samples)] = corr[n];
			mf[n / 2 + (2 * k + 1) * (2 * n_samples)] = corr[n + 1];
		}
	}
	fclose(fp);
	free(p);
	free(pulse);
	free(corr);

	/* create arrays for FFT */

	double *q = malloc(2 * m * sizeof(double));
	double *r = malloc(m * sizeof(double));
	double *l = malloc(2 * m * sizeof(double));
	double *f = malloc(m * (2 * n_samples) * sizeof(double));
	double max = 0, a, b;

	/* FFT */

	for (x = 0; x < 2 * n_samples; x++) {
		for (o = 0; o < 2 * m; o++) {
			l[o] = mf[x + o * (2 * n_samples)];
		}
        KERN_ENTER(make_label("FFT[1D][forward][complex][float64][%d]",m));
        gsl_fft(l,q,m);
        KERN_EXIT(make_label("FFT[1D][forward][complex][float64][%d]",m));
		for (y = 0; y < 2 * m; y += 2) {
			r[y / 2] = sqrt(q[y] * q[y] + q[y + 1] * q[y + 1]);  // calculate the absolute value of the output
		}
		fftshift(r, m);

		for (z = 0; z < m; z++) {
			f[x + z * (2 * n_samples)] = r[z];  // put the elements of output into corresponding location of the 2D
			                                    // array
			if (r[z] > max) {
				max = r[z];
				a = z + 1;
				b = x + 1;
			}
		}
	}
	fp = fopen(OUTPUT, "w");  // write the output
	for (i = 0; i < m; i++) {
		for (j = 0; j < 2 * n_samples; j++) {
			fprintf(fp, "%lf ", f[j + i * (2 * n_samples)]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	free(mf);
	free(q);
	free(r);
	free(l);
	double rg, dp;
	rg = (b - n_samples) / (n_samples - 1) * PRI;
	dp = (a - (m + 1) / 2) / (m - 1) / PRI;
	printf("doppler shift = %lf, time delay = %lf", dp, rg);
}
