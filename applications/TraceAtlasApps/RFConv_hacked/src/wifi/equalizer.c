#include <stdio.h>
#include <string.h>

#include "channel_Eq.h"
#include "channel_Eq_initialize.h"
#include "channel_Eq_terminate.h"
#include "common.h"

static float pilot_data[32] = {-1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1,
                               -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1};
static creal_T dftm[4096];

void init_equalizer() {
	// DASH_DATA
	if (!getenv("DASH_DATA")) {
		printf("in equalizer.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char* file0 = "Dash-RadioCorpus/wifi/f_complex.dat";
	char* path = (char*)malloc(FILEPATH_SIZE * sizeof(char));
	strcat(path, getenv("DASH_DATA"));
	strcat(path, file0);
	FILE* fp = fopen(path, "rb");

	if (fp == NULL) {
		printf("in equalizer.c:\n\tFATAL: %s not found!\n", file0);
		exit(1);
	}

	int i, j;
	float dftm_r[64 * 64 * 2];

	// DFTmatrix
	fread(&dftm_r, sizeof(dftm_r), 1, fp);
	fclose(fp);

	for (i = 0, j = 0; i < 4096; i++) {
		dftm[j].re = dftm_r[i];
		dftm[i].im = dftm_r[i + 4096];
		j++;
	}
}

void equalization(float pilotdata_rx[], float offt[], float out_fft_c2f[], int mode) {
	creal_T tpilot[16];
	creal_T rpilot[16];
	creal_T fftout[128];
	creal_T eqData[128];

	f2c(32, pilot_data, tpilot);
	f2c(32, pilotdata_rx, rpilot);
	f2c(128 * 2, offt, fftout);

	//    channel_Eq(tpilot, rpilot, fftout, dftm, eqData, mode);

	c2f(FFT_N, eqData, out_fft_c2f);
}
