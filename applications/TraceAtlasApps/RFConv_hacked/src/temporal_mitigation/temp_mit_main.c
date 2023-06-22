#define _GNU_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>

#include "inverse.h"
#include "common.h"

void *temp_mit(){
		// DASH_DATA
		if (!getenv("DASH_DATA")) {
			printf("in TX.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
			exit(1);
		}
		
		char *file = "Dash-RadioCorpus/QPR8_RFConvSys/txdata_remod.txt";
		char *path = (char *)malloc(FILEPATH_SIZE * sizeof(char));
		strcat(path, getenv("DASH_DATA"));
		strcat(path, file);
		FILE *cfp = fopen(path, "r");
		free(path);
		
		if (cfp == NULL) {
		printf("in main.c:\n\tFATAL: %s was not found!", file);
		exit(1);
		}

		double temp_real, temp_imag;
		int num_tx_samp = 64;

		float *CommsReal, *CommsImag;
		CommsReal = (float *)malloc(4 * num_tx_samp * sizeof(float));
		CommsImag = (float *)malloc(4 * num_tx_samp * sizeof(float));
		
		float *RxReal, *RxImag;
		RxReal = (float *)malloc(4 * num_tx_samp * sizeof(float));
		RxImag = (float *)malloc(4 * num_tx_samp * sizeof(float));
		
		float *RadarReal, *RadarImag;
		RadarReal = (float *)malloc(4 * num_tx_samp * sizeof(float));
		RadarImag = (float *)malloc(4 * num_tx_samp * sizeof(float));
	
		gsl_matrix_complex *wifi_tx = NULL;	
		wifi_tx = gsl_matrix_complex_alloc(1, num_tx_samp);		
		gsl_complex temp_complex = gsl_complex_rect(1.,0.);
		gsl_complex complexZero = gsl_complex_rect(0.,0.);

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < num_tx_samp; j++) {
				fscanf(cfp, "%f %f\n", &temp_real, &temp_imag);
				temp_complex = gsl_complex_rect(temp_real, temp_imag);
				gsl_matrix_complex_set(wifi_tx, i, j, temp_complex);
			}
		}
		fclose(cfp);
		
		char *file0 = "Dash-RadioCorpus/QPR8_RFConvSys/rxdata.txt";
		char *path0 = (char *)malloc(FILEPATH_SIZE * sizeof(char));
		strcat(path0, getenv("DASH_DATA"));
		strcat(path0, file0);
		FILE *cfp0 = fopen(path0, "r");
		free(path0);
		
		if (cfp0 == NULL) {
			printf("in main.c:\n\tFATAL: %s was not found!", file0);
			exit(1);
		}

		for (int j = 0; j < num_tx_samp; j++) {
			fscanf(cfp0, "%f %f\n", &temp_real, &temp_imag);
			RxReal[0 * num_tx_samp + j] = temp_real;
			RxImag[0 * num_tx_samp + j] = temp_imag;
			RxReal[1 * num_tx_samp + j] = temp_real;
			RxImag[1 * num_tx_samp + j] = temp_imag;
			RxReal[2 * num_tx_samp + j] = temp_real;
			RxImag[2 * num_tx_samp + j] = temp_imag;
			RxReal[3 * num_tx_samp + j] = temp_real;
			RxImag[3 * num_tx_samp + j] = temp_imag;
			}		
		fclose(cfp0);

		gsl_matrix_complex *S_temp_delay = NULL;
		S_temp_delay = gsl_matrix_complex_alloc(4, num_tx_samp);  // should be 4x64

		//set the appropriate zeros
		gsl_matrix_complex_set(S_temp_delay, 0, num_tx_samp - 1, complexZero);  // always the case for all blocks
		gsl_matrix_complex_set(S_temp_delay, 2, 0, complexZero);             // always the case for all blocks
		gsl_matrix_complex_set(S_temp_delay, 3, 0, complexZero);             // always the case for all blocks
		gsl_matrix_complex_set(S_temp_delay, 3, 1, complexZero);             // always the case for all blocks


		// The -1_th delay tap
		for (int i = 0; i < num_tx_samp - 1; i++) {
			gsl_matrix_complex_set(S_temp_delay, 0, i, gsl_matrix_complex_get(wifi_tx, 0, i + 1));
		}
		// Now the LOS and other delayed taps
		for (int i = 1; i < 4; i++) {
			for (int j = 0; j < num_tx_samp - (i - 1); j++) {
				gsl_matrix_complex_set(S_temp_delay, i, j + (i - 1),
				gsl_matrix_complex_get(wifi_tx, 0, j));
			}
		}
		// Convert GSL matrix to regular arrays for temporal mitigation
		for (int i = 1; i < 4; i++) {
			for (int j = 0; j < num_tx_samp; j++) {
				CommsReal[i * num_tx_samp + j] = GSL_REAL(gsl_matrix_complex_get(S_temp_delay,i,j));
				CommsImag[i * num_tx_samp + j] = GSL_IMAG(gsl_matrix_complex_get(S_temp_delay,i,j));
			}
		}

		for (int i = 1; i < 4; i++) {
			for (int j = 0; j < num_tx_samp; j++) {
				CommsReal[i * num_tx_samp + j] = GSL_REAL(gsl_matrix_complex_get(S_temp_delay,i,j));
				CommsImag[i * num_tx_samp + j] = GSL_IMAG(gsl_matrix_complex_get(S_temp_delay,i,j));
			}
		}
		
		temporalmitigation(CommsReal,CommsImag,RxReal,RxImag,RadarReal,RadarImag);

		char *file1 = "Dash-RadioCorpus/QPR8_RFConvSys/received_input.txt";
		char *path1 = (char *)malloc(FILEPATH_SIZE * sizeof(char));
		strcat(path1, getenv("DASH_DATA"));
		strcat(path1, file1);
		FILE *cfp1 = fopen(path1, "w");
		free(path1);

		if (cfp1 == NULL) {
		printf("in main.c:\n\tFATAL: %s was not found!", file1);
		exit(1);
		}


		for (int j = 0; j < num_tx_samp; j++) {
				fscanf(cfp, "%f %f\n", &temp_real, &temp_imag);
				RxReal[0 * num_tx_samp + j] = temp_real;
				RxImag[0 * num_tx_samp + j] = temp_imag;
				RxReal[1 * num_tx_samp + j] = temp_real;
				RxImag[1 * num_tx_samp + j] = temp_imag;
				RxReal[2 * num_tx_samp + j] = temp_real;
				RxImag[2 * num_tx_samp + j] = temp_imag;
				RxReal[3 * num_tx_samp + j] = temp_real;
				RxImag[3 * num_tx_samp + j] = temp_imag;
		}

		for (int i = 0; i < num_tx_samp; i++) {
			fprintf(cfp1, "%f %f\n", (RadarReal[0 * num_tx_samp + i] + RadarReal[1 * num_tx_samp + i] + RadarReal[2 * num_tx_samp + i] + RadarReal[3 * num_tx_samp + i]), (RadarImag[0 * num_tx_samp + i] + RadarImag[1 * num_tx_samp + i] + RadarImag[2 * num_tx_samp + i] + RadarImag[3 * num_tx_samp + i]));
		}

		fclose(cfp1);
}
