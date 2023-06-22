//
//  RF_convergence.c
//
//  Created by Matthew Kinsinger (Student) on 2/20/20.
//
//  Combining already written applications of sync_MMSE_beamformer (Matt), STAP_Comms (Matt),
//  and temporal_mitigation (Saquib) to make a full (simplistic) RF convergence simulation
//  in which both a radar and communications signal are overlapping in time and frequency.
//  We sync to the comms signal, MMSE estimate, decode and remodulate, then temporally project
//  the full received data onto the subspace orthogonal to the received comms signal.
//
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include<iostream>
//#include<fstream>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <math.h>

#include "../../include/DashExtras.h"
#include "../../include/gsl_fft_mex.c"
#include "../../include/gsl_ifft_mex.c"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/RF_convergence/simple_RF_convergence"
#define RX PROGPATH "rx.dat"
#define PDPULSE PROGPATH "input_pd_pulse.txt"
//#define DATAEST PROGPATH "dataEstimate.txt"
#define XTRAIN PROGPATH "x_Training.dat"
#define SYNCTRAIN PROGPATH "sync_symbols.dat"

// Prototypes of the functions to be called by main()

// returns the starting time sample index for the comms signal (ipeak = 400)
int sync_MMSE(int numRx, int numTx, int Ntaps, int num_sync_samp, int num_rx_samp, double *sync_symbols_real,
              double *sync_symbols_imag, double *rx_data_sync_real, double *rx_data_sync_imag);
void LU_factorization_projection(gsl_matrix_complex *Z_temp_proj, int numRx, gsl_matrix_complex *S_temp_delay,
                                 int Ntaps_projection, int modulo_N);
void xcorr(double *x, double *y, size_t n_samp, double *corr);

int main() {
	// Read in data structures and parameters
	int numRx = 4;  // ALWAYS KEEP AS 4!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation code)
	int numTx = 1;  // ALWAYS KEEP AS 1!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation code)
	int Ntaps = 5;  // for sync and MMSE beamforming
	int initial_offset = -1;   // begin at the -1th delay tap for the STAP data matrix
	int Ntaps_projection = 4;  // ALWAYS KEEP AS 4!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation
	                           // code)
	// error checking
	if (numRx != Ntaps_projection) {
		printf(
		    "Number of receive antennas needs to be equal to Ntaps for the projection matrix (both equal to 4).\r\n");
		printf("numRx: %d, Ntaps_projection: %d\r\n", numRx, Ntaps_projection);
		exit(4);
	}
	int modulo_N = 64;        // We need to call Saquib's temporal mitigation code in blocks of 64 time samples
	int n_sync_blocks = 4;    // number of comms sync blocks of 64 time samples
	int n_trn_blocks = 4;     // number of comms training blocks of 64 time samples
	int n_data_blocks = 2030;//280;  // number of comms data blocks of 64 time samples
	int n_start_zeros = 400;
	int num_sync_samp = n_sync_blocks * modulo_N;
	int n_sync_trn_zeros = 100;  // zero padding between comms sync and training signals
	int num_trn_samp = n_trn_blocks * modulo_N;
	int num_trn_data_zeros = 100;  // zero padding between comms training and data signals
	int num_data_samp = n_data_blocks * modulo_N;
	int n_data_end_zeros = 40;//18;  // zero padding at the end of the comms signal. This number needs to be hard coded so
	                            // that we have total time samples of 19050
	int num_rx_samp = 131072;//19050;
	// error check
	int samp_sum = n_start_zeros + num_sync_samp + n_sync_trn_zeros + num_trn_samp + num_trn_data_zeros +
	               num_data_samp + n_data_end_zeros;
	if (samp_sum != num_rx_samp) {
		printf("Number of total Rx time samples is not correct?\r\n");
		printf("Sum of samples: %d, N Rx samples: %d\r\n", samp_sum, num_rx_samp);
		exit(4);
	}
	// QAM comms constellation parameters (hard coded for now)
	int N_constellation_points = 4;
	gsl_matrix_complex *QAM_constellation = NULL;
	QAM_constellation = gsl_matrix_complex_alloc(1, N_constellation_points);
	gsl_matrix_complex_set(QAM_constellation, 0, 0, gsl_complex_rect(1., 0.));
	gsl_matrix_complex_set(QAM_constellation, 0, 1, gsl_complex_rect(0., 1.));
	gsl_matrix_complex_set(QAM_constellation, 0, 2, gsl_complex_rect(-1., 0.));
	gsl_matrix_complex_set(QAM_constellation, 0, 3, gsl_complex_rect(0., -1.));

	/*
	 * Read in raw data - and form into signal matrix
	 */
	gsl_matrix_complex *rxSig = NULL;
	FILE *rawDataFile = fopen(RX, "r");
	// FILE *rawDataFile = fopen(RX,"r");
	rxSig = gsl_matrix_complex_alloc(numRx, num_rx_samp);
	double temp_real, temp_imag;
	gsl_complex temp_complex = gsl_complex_rect(1., 0.);
	int dataNumPulses;
	fread(&dataNumPulses, sizeof(int), 1, rawDataFile);
	if ((numRx * num_rx_samp) != dataNumPulses) {
		printf(
		    "Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
		printf("dataNumPulses: %d, numPulses: %d\r\n", dataNumPulses, numRx * num_rx_samp);
		exit(4);
	}
	KERN_ENTER(make_label("fileLoad[2D][%d,%d][complex][float64]",numRx,num_rx_samp));
    for (int i = 0; i < num_rx_samp; i++) {
		for (int j = 0; j < numRx; j++) {
			fread(&temp_real, sizeof(double), 1, rawDataFile);
			fread(&temp_imag, sizeof(double), 1, rawDataFile);
			temp_complex = gsl_complex_rect(temp_real, temp_imag);
			gsl_matrix_complex_set(rxSig, j, i, temp_complex);
		}
	}
	KERN_EXIT(make_label("fileLoad[2D][%d,%d][complex][float64]",numRx,num_rx_samp));
	fclose(rawDataFile);

	//     //debugging: check to see if we read the data in correctly (done)
	//     {
	//         FILE * f = fopen ("readInRxSig.txt", "w");
	//         gsl_matrix_complex_fprintf(f,rxSig,"%f");
	//         fclose(f);
	//     }

	printf("Done reading in received signal.\r\n");

	/*
	 * Read in sync data
	 */
	gsl_matrix_complex *sync_symbols = NULL;
	FILE *rawDataFile2 = fopen(SYNCTRAIN, "r");
	// FILE *rawDataFile2 = fopen(SYNCTRAIN,"r");
	sync_symbols = gsl_matrix_complex_alloc(numTx, num_sync_samp);
	fread(&dataNumPulses, sizeof(int), 1, rawDataFile2);
	if ((num_sync_samp) != dataNumPulses) {
		printf(
		    "Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
		printf("dataNumPulses: %d, numPulses: %d\r\n", dataNumPulses, num_sync_samp);
		exit(4);
	}
    KERN_ENTER(make_label("fileLoad[1D][%d][complex][float64]",num_sync_samp));
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < num_sync_samp; j++) {
			fread(&temp_real, sizeof(double), 1, rawDataFile2);
			fread(&temp_imag, sizeof(double), 1, rawDataFile2);
			temp_complex = gsl_complex_rect(temp_real, temp_imag);
			gsl_matrix_complex_set(sync_symbols, i, j, temp_complex);
		}
	}
    KERN_EXIT(make_label("fileLoad[1D][%d][complex][float64]",num_sync_samp));
	fclose(rawDataFile2);
	printf("Done reading in sync reference.\r\n");

	//     //debugging: check to see if we read the data in correctly (done)
	//     {
	//         FILE * f = fopen ("readInX_training.txt", "w");
	//         gsl_matrix_complex_fprintf(f,x_Training,"%f");
	//         fclose(f);
	//     }

	/*
	 * Read in comms training data
	 */
	gsl_matrix_complex *x_Training = NULL;
	FILE *rawDataFile3 = fopen(XTRAIN, "r");
	// FILE *rawDataFile3 = fopen(XTRAIN,"r");
	x_Training = gsl_matrix_complex_alloc(1, num_trn_samp);
	fread(&dataNumPulses, sizeof(int), 1, rawDataFile3);
	if ((num_trn_samp) != dataNumPulses) {
		printf(
		    "Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
		printf("dataNumPulses: %d, numPulses: %d\r\n", dataNumPulses, num_trn_samp);
		exit(4);
	}
    KERN_ENTER(make_label("fileLoad[1D][%d][complex][float64]",num_trn_samp));
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < num_trn_samp; j++) {
			fread(&temp_real, sizeof(double), 1, rawDataFile2);
			fread(&temp_imag, sizeof(double), 1, rawDataFile2);
			temp_complex = gsl_complex_rect(temp_real, temp_imag);
			gsl_matrix_complex_set(x_Training, i, j, temp_complex);
		}
	}
    KERN_EXIT(make_label("fileLoad[1D][%d][complex][float64]",num_trn_samp));
	fclose(rawDataFile2);
	printf("Done reading in training reference.\r\n");

	//     //debugging: check to see if we read the data in correctly (done)
	//     {
	//         FILE * f = fopen ("readInX_training.txt", "w");
	//         gsl_matrix_complex_fprintf(f,x_Training,"%f");
	//         fclose(f);
	//     }

	// =================== begin synchronization to the comms signal ======================================
	// Get real and imaginary parts of the complex matrices (both sync and rx data)
	// Dynamic memory initialization
	double *sync_symbols_real, *sync_symbols_imag;
	double *rx_data_sync_real, *rx_data_sync_imag;
	// Dynamic memory allocation
	sync_symbols_real = (double *)malloc(numTx * num_sync_samp * sizeof(double));
	sync_symbols_imag = (double *)malloc(numTx * num_sync_samp * sizeof(double));

	rx_data_sync_real = (double *)malloc(numRx * num_rx_samp * sizeof(double));
	rx_data_sync_imag = (double *)malloc(numRx * num_rx_samp * sizeof(double));

	for (int i = 0; i < numTx; i++) {
		for (int j = 0; j < num_sync_samp; j++) {
			// QUESTION: Does this dynamic memory access automatically account for the size of the data type?
			sync_symbols_real[i * num_sync_samp + j] = GSL_REAL(gsl_matrix_complex_get(sync_symbols, i, j));
			sync_symbols_imag[i * num_sync_samp + j] = GSL_IMAG(gsl_matrix_complex_get(sync_symbols, i, j));
		}
	}

	for (int i = 0; i < numRx; i++) {
		for (int j = 0; j < num_rx_samp; j++) {
			rx_data_sync_real[i * num_rx_samp + j] = GSL_REAL(gsl_matrix_complex_get(rxSig, i, j));
			rx_data_sync_imag[i * num_rx_samp + j] = GSL_IMAG(gsl_matrix_complex_get(rxSig, i, j));
		}
	}

	// get the time index at which the comms signal begins
	int ipeak = sync_MMSE(numRx, numTx, Ntaps, num_sync_samp, num_rx_samp, sync_symbols_real, sync_symbols_imag,
	                      rx_data_sync_real, rx_data_sync_imag);

	// =================== end sync to the comms signal ======================================

	// =================== begin MMSE comms signal estimation ================================

	// beginning at the first comms training sample we build the Space-time receive matrix
	int comms_sync_start_idx = ipeak;
	int comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros;
	int comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;

	// Make the delayed matrix, rxSigDelays (-1 + LOS + 3 delay taps, say)
	int stackedMatSize = Ntaps * numRx;
	double zero = 0;
	gsl_complex complexZero = gsl_complex_rect(zero, zero);
	// Slide rows of rxSig matrix to the left by 1 sample. Fill in with zeros at the end
	gsl_matrix_complex *rxSigDelays = NULL;
	rxSigDelays = gsl_matrix_complex_alloc(stackedMatSize, num_trn_samp);  // allocate memory for the space-time matrix
	// fill in the space-time matrix using rxSig. Zero pad at the end.
	for (int i = 0; i < numRx; i++) {             // antenna loop
		for (int j = 0; j < num_trn_samp; j++) {  // time samples loop
			// place the samples for antenna i for LOS and all delays
			for (int k = initial_offset; k < Ntaps + initial_offset; k++) {  // delay taps loop
				// printf("I: %d, J: %d, K: %d\r\n",i,j,k);
				if (j + k <= num_trn_samp - 1 + initial_offset) {
					temp_complex = gsl_matrix_complex_get(rxSig, i, comms_trn_start_idx + j + k);  // the k-th delay
					                                                                               // sample
					gsl_matrix_complex_set(rxSigDelays, numRx * (k - initial_offset) + i, j, temp_complex);
				} else {
					gsl_matrix_complex_set(rxSigDelays, numRx * (k - initial_offset) + i, j, complexZero);  // set
					                                                                                        // samples
					                                                                                        // at the
					                                                                                        // end to
					                                                                                        // 0+i0
				}
			}
		}
	}

	//     //debugging: check to see if we buit STAP matrix correctly (done)
	//     {
	//         FILE * f = fopen ("rxSigDelays.txt", "w");
	//         gsl_matrix_complex_fprintf(f,rxSigDelays,"%f");
	//         fclose(f);
	//     }

	//    gsl_matrix_complex_transpose_memcpy(rxSigDelays,rxSig);

	// Now make the cross-correlation matrix (rxSigDelays * rxSigDelaysHerm)
	gsl_matrix_complex *autoCorrMatrix = NULL;
	autoCorrMatrix = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);  // allocate memory
	gsl_complex sampleToConjugate = gsl_complex_rect(1., 0.);  // To be used to hold the sample to be conjugate in each
	                                                           // loop
	gsl_complex dot_prod = gsl_complex_rect(0., 0.);           // declare type for the dot product accumulator
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_trn_samp,stackedMatSize));
	for (int i = 0; i < stackedMatSize; i++) {
		for (int k = 0; k < stackedMatSize; k++) {
			// dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
			dot_prod = gsl_complex_rect(0., 0.);
			for (int j = 0; j < num_trn_samp; j++) {
				temp_complex = gsl_matrix_complex_get(rxSigDelays, i, j);
				sampleToConjugate = gsl_matrix_complex_get(rxSigDelays, k, j);  // This is the sample to conjugate
				sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
				temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate);  // Multiply the two samples
				dot_prod = gsl_complex_add(dot_prod, temp_complex);               // accumulate (dot product)
			}
			// Place this dot product into the cross correlation matrix
			gsl_matrix_complex_set(autoCorrMatrix, i, k, dot_prod);
		}
	}
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_trn_samp,stackedMatSize));
	// end making the auto-correlation matrix

	//     //debugging: check to see if we buit auto-correlation matrix correctly (done)
	//     {
	//         FILE * f = fopen ("autoCorrMatrix.txt", "w");
	//         gsl_matrix_complex_fprintf(f,autoCorrMatrix,"%f");
	//         fclose(f);
	//     }

	// Invert the auto-correlation matrix
	gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
	int s;
	// Compute the LU decomposition of this matrix
	printf("just before LU decomposition\n");

	KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));
	gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
	// Compute the  inverse of the LU decomposition
	gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
	printf("just before inversion\n");
	gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
	KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));
	
	printf("just after inversion\n");
	gsl_permutation_free(p);

	//     //debugging: check to see if we inverted the auto-correlation matrix correctly (done)
	//     {
	//         FILE * f = fopen ("invAutoCorr.txt", "w");
	//         gsl_matrix_complex_fprintf(f,invAutoCorr,"%f");
	//         fclose(f);
	//     }

	// Matrix multiply arrayResponse = rxSigDelayed * x_training^H
	gsl_matrix_complex *arrayResponse = NULL;
	arrayResponse = gsl_matrix_complex_alloc(stackedMatSize, 1);  // allocate memory
	// complex matrix multiplication
	gsl_complex unity = gsl_complex_rect(1., 0.);  // the complex number 1
	printf("just before making array response.\n");

	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_trn_samp));
	gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, rxSigDelays, x_Training, complexZero, arrayResponse);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_trn_samp));

	//     //debugging: check to see if we built the array response vector correctly (done)
	//     {
	//         FILE * f = fopen ("arrayResponse.txt", "w");
	//         gsl_matrix_complex_fprintf(f,arrayResponse,"%f");
	//         fclose(f);
	//     }

	printf("just before making beamormer.\n");
	// Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
	// This gives us our beamforming vector w (20x1)
	gsl_matrix_complex *beamFormer = NULL;
	beamFormer = gsl_matrix_complex_alloc(stackedMatSize, 1);  // allocate memory
	// complex matrix multiplication

	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complexZero, beamFormer);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));

	printf("just after making beamormer.\n");

	//     //debugging: check to see if we built the beamformer vector correctly (done)
	//     {
	//         FILE * f = fopen ("beamFormer.txt", "w");
	//         gsl_matrix_complex_fprintf(f,beamFormer,"%f");
	//         fclose(f);
	//     }

	// Make the delayed matrix, rxSigDelays (LOS + 4 delay taps, say)
	// Slide rows of rxSig matrix to the left by 1 sample. Fill in with zeros at the end

	int Nextra = -initial_offset;  // we still need to make sure we grab all the way to the end of the data sequence
	                               // length
	gsl_matrix_complex *rxDataDelays = NULL;
	rxDataDelays = gsl_matrix_complex_alloc(stackedMatSize, num_data_samp + Nextra);  // allocate memory for the
	                                                                                  // space-time matrix
	// fill in the space-time matrix using rxSig
	for (int i = 0; i < numRx; i++) {                       // antenna loop
		for (int j = 0; j < num_data_samp + Nextra; j++) {  // time samples loop
			// place the samples for antenna i for LOS and all delays
			for (int k = initial_offset; k < Ntaps + initial_offset; k++) {  // delay taps loop beginning at -1
				//               printf("I: %d, J: %d, K: %d\r\n",i,j,k);
				if (j + k < num_data_samp - 1 + initial_offset + Nextra) {
					temp_complex = gsl_matrix_complex_get(rxSig, i, comms_data_start_idx + j + k);  // the k-th delay
					                                                                                // sample
					gsl_matrix_complex_set(rxDataDelays, numRx * (k - initial_offset) + i, j, temp_complex);
				} else {
					gsl_matrix_complex_set(rxDataDelays, numRx * (k - initial_offset) + i, j, complexZero);  // set
					                                                                                         // samples
					                                                                                         // at the
					                                                                                         // end to
					                                                                                         // 0+i0
				}
			}
		}
	}

	printf("Built delayed data matrix.\r\n");

	// Multiply w^H * rxDataDelayed
	gsl_matrix_complex *dataEstimate = NULL;
	dataEstimate = gsl_matrix_complex_alloc(1, num_data_samp + Nextra);  // allocate memory
	// complex matrix multiplication
	KERN_ENTER(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_data_samp + Nextra));
	gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, rxDataDelays, complexZero, dataEstimate);
	KERN_EXIT(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_data_samp + Nextra));

	// MAKE SURE TO DISREGARD THE LAST <Nextra> ENTRY'S OF <dataEstimate> IN ANY FURTHER PROCESSING (i.e. when building
	// the projection matrix)

	printf("Formed beamformed data estimate.\r\n");
	// This gives us our estimate of the transmitted signal (1xnumDataSamples?)
	//     // Write out estimated signal to file.
	//     {
	//         FILE * f = fopen ("data_estimate.txt", "w");
	//         gsl_matrix_complex_fprintf(f,dataEstimate,"%f");
	//         fclose(f);
	//     }

	// Now need to map the estimate of what was transmitted back to QAM symbols by
	// finding the QAM symbol that is nearest to the estimated symbol
	// QAM_constellation 1x4
	// dataEstimate 1xnum_data_samp    (make sure to disregard the last Nextra (=1) entries

	// gsl_matrix_complex *remodulated_symbols = NULL;
	// remodulated_symbols = gsl_matrix_complex_alloc(1,num_data_samp);

	// make a vector to temporarily hold the distance metrics
	double temp_dist[N_constellation_points];
	int min_idx;
	gsl_complex temp_complex2 = gsl_complex_rect(1., 0.);

	KERN_ENTER(make_label("ZIP[4QAM_argmin][%d][float64][complex]",num_data_samp*N_constellation_points));
	for (int i = 0; i < num_data_samp; i++) {
		temp_complex = gsl_matrix_complex_get(dataEstimate, 0, i);
		for (int j = 0; j < N_constellation_points; j++) {
			temp_complex2 = gsl_matrix_complex_get(QAM_constellation, 0, j);
			temp_complex2 = gsl_complex_sub(temp_complex, temp_complex2);
			temp_dist[j] = gsl_complex_abs2(temp_complex2);
		}
		// find minimum entry in temp_dist
		min_idx = 0;
		for (int k = 1; k < N_constellation_points; k++) {
			if (temp_dist[k] < temp_dist[min_idx]) {
				min_idx = k;
			}
		}
		// quanitze the dataEstimate to the nearest QAM symbol
		gsl_matrix_complex_set(dataEstimate, 0, i, gsl_matrix_complex_get(QAM_constellation, 0, min_idx));
	}
	KERN_EXIT(make_label("ZIP[4QAM_argmin][%d][float64][complex]",num_data_samp*N_constellation_points));

	printf("Remodulated the data estimate.\r\n");

	// Print out the remodulated data estimate to check number of symbol errors (for debugging in Matlab)
	//         {
	//             FILE * f = fopen ("data_remodulated.txt", "w");
	//             gsl_matrix_complex_fprintf(f,dataEstimate,"%f");
	//             fclose(f);
	//         }

	// =================== end MMSE comms signal estimation ================================

	// =================== begin orthogonal projection =====================================
	// Now we need to project the received signal onto the subspace orthogonal to the comms signal.
	// Beginning at the sync index we project the sync signal, trn signal, and comms data signal.
	// IGNORE the sections between these signal where we zero padded the comms signal.

	//=============== BEGIN sync symbols section =========================================

	gsl_matrix_complex *S_temp_delay = NULL;
	S_temp_delay = gsl_matrix_complex_alloc(numTx * Ntaps_projection, modulo_N);  // 4x64

	gsl_matrix_complex *Z_temp_proj = NULL;
	Z_temp_proj = gsl_matrix_complex_alloc(numRx, modulo_N);  // 4x64

	// gsl_matrix_complex_set(dataEstimate,0,i,gsl_matrix_complex_get(QAM_constellation,0,min_idx));

	gsl_matrix_complex_set(S_temp_delay, 0, modulo_N - 1, complexZero);  // always the case for all blocks
	gsl_matrix_complex_set(S_temp_delay, 2, 0, complexZero);             // always the case for all blocks
	gsl_matrix_complex_set(S_temp_delay, 3, 0, complexZero);             // always the case for all blocks
	gsl_matrix_complex_set(S_temp_delay, 3, 1, complexZero);             // always the case for all blocks

	// Now the non-zero entries (one block of 64 samples at a time)
	for (int k = 0; k < n_sync_blocks; k++) {
		// The -1_th delay tap
		for (int i = 0; i < modulo_N - 1; i++) {
			gsl_matrix_complex_set(S_temp_delay, 0, i, gsl_matrix_complex_get(sync_symbols, 0, k * modulo_N + i + 1));
		}
		// Now the LOS and other delayed taps
		for (int i = 1; i < Ntaps_projection; i++) {
			for (int j = 0; j < modulo_N - (i - 1); j++) {
				gsl_matrix_complex_set(S_temp_delay, i, j + (i - 1),
				                       gsl_matrix_complex_get(sync_symbols, 0, k * modulo_N + j));
			}
		}
		// Now grab the relevant received data block
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_sync_start_idx = ipeak;
				gsl_matrix_complex_set(Z_temp_proj, i, j,
				                       gsl_matrix_complex_get(rxSig, i, comms_sync_start_idx + k * modulo_N + j));
			}
		}
		// Now project using this block. After calling the function the matrix Z_temp_proj will hold the receivd data
		// after projection onto the subspace orthogonal to S_temp_delay.
		LU_factorization_projection(Z_temp_proj, numRx, S_temp_delay, Ntaps_projection, modulo_N);
		// Replace the corresponding section of the received data with this newly projected data
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_sync_start_idx = ipeak;
				gsl_matrix_complex_set(rxSig, i, comms_sync_start_idx + k * modulo_N + j,
				                       gsl_matrix_complex_get(Z_temp_proj, i, j));
			}
		}
	}
	//=============== END sync symbols section =========================================

	//=============== BEGIN training symbols section =========================================

	// Now the non-zero entries (one block of 64 samples at a time)
	for (int k = 0; k < n_trn_blocks; k++) {
		// The -1_th delay tap
		for (int i = 0; i < modulo_N - 1; i++) {
			gsl_matrix_complex_set(S_temp_delay, 0, i, gsl_matrix_complex_get(x_Training, 0, k * modulo_N + i + 1));
		}
		// Now the LOS and other delayed taps
		for (int i = 1; i < Ntaps_projection; i++) {
			for (int j = 0; j < modulo_N - (i - 1); j++) {
				gsl_matrix_complex_set(S_temp_delay, i, j + (i - 1),
				                       gsl_matrix_complex_get(x_Training, 0, k * modulo_N + j));
			}
		}
		// Now grab the relevant received data block
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros;
				gsl_matrix_complex_set(Z_temp_proj, i, j,
				                       gsl_matrix_complex_get(rxSig, i, comms_trn_start_idx + k * modulo_N + j));
			}
		}
		// Now project using this block. After calling the function the matrix Z_temp_proj will hold the receivd data
		// after projection onto the subspace orthogonal to S_temp_delay.
		LU_factorization_projection(Z_temp_proj, numRx, S_temp_delay, Ntaps_projection, modulo_N);
		// Replace the corresponding section of the received data with this newly projected data
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_sync_start_idx = ipeak;
				gsl_matrix_complex_set(rxSig, i, comms_trn_start_idx + k * modulo_N + j,
				                       gsl_matrix_complex_get(Z_temp_proj, i, j));
			}
		}
	}
	//=============== END training symbols section =========================================

	//=============== BEGIN data symbols section =========================================

	// Now the non-zero entries (one block of 64 samples at a time)
	for (int k = 0; k < n_data_blocks; k++) {
		// The -1_th delay tap
		for (int i = 0; i < modulo_N - 1; i++) {
			gsl_matrix_complex_set(S_temp_delay, 0, i, gsl_matrix_complex_get(dataEstimate, 0, k * modulo_N + i + 1));
		}
		// Now the LOS and other delayed taps
		for (int i = 1; i < Ntaps_projection; i++) {
			for (int j = 0; j < modulo_N - (i - 1); j++) {
				gsl_matrix_complex_set(S_temp_delay, i, j + (i - 1),
				                       gsl_matrix_complex_get(dataEstimate, 0, k * modulo_N + j));
			}
		}
		// Now grab the relevant received data block
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;
				gsl_matrix_complex_set(Z_temp_proj, i, j,
				                       gsl_matrix_complex_get(rxSig, i, comms_data_start_idx + k * modulo_N + j));
			}
		}
		// Now project using this block. After calling the function the matrix Z_temp_proj will hold the receivd data
		// after projection onto the subspace orthogonal to S_temp_delay.
		LU_factorization_projection(Z_temp_proj, numRx, S_temp_delay, Ntaps_projection, modulo_N);
		// Replace the corresponding section of the received data with this newly projected data
		for (int i = 0; i < numRx; i++) {
			for (int j = 0; j < modulo_N; j++) {
				// int comms_sync_start_idx = ipeak;
				gsl_matrix_complex_set(rxSig, i, comms_data_start_idx + k * modulo_N + j,
				                       gsl_matrix_complex_get(Z_temp_proj, i, j));
			}
		}
	}
	//=============== END data symbols section =========================================

	printf("Done projecting onto subspace orthogonal to comms signal.");
	// =================== END orthogonal projection =====================================

	//=============== BEGIN radar processing section =========================================
	
	size_t n_samples = 1024;
	double B = 1e6;
	double T = 100/B;
	double sampling_rate = 1e6;
	double lag;
	double *corr = malloc((2 * (2 * n_samples - 1)) * sizeof(double));
	double *received = malloc(2 * n_samples * sizeof(double));
	double *pulse = malloc(2 * n_samples * sizeof(double));         // array for the original pulse
	
	for (int i = 0; i < n_samples; i++) {
		temp_complex = gsl_complex_add(gsl_matrix_complex_get(rxSig, 0, i),gsl_complex_add(gsl_matrix_complex_get(rxSig, 1, i), gsl_complex_add(gsl_matrix_complex_get(rxSig, 2, i),gsl_matrix_complex_get(rxSig, 3, i))));
		received[2*i] = GSL_REAL(temp_complex);
		received[2*i + 1] = GSL_IMAG(temp_complex);
	}

    FILE *fp;
	fp = fopen(PDPULSE, "r");  // read the original pulse
	for (int i = 0; i < 2 * n_samples; i++) {
		fscanf(fp, "%lf", &pulse[i]);
	}
	fclose(fp);
	

	xcorr(received, pulse, n_samples, corr);

	// Code to find maximum
	double max_corr = 0,tmp=0;
	double index = 0;
	for (size_t i = 0; i < 2 * (2 * n_samples - 1); i += 2) {
		// Only finding maximum of real part of correlation
		tmp = corr[i]*corr[i] + corr[i+1]*corr[i+1];
		if (corr[i] > max_corr) {
			max_corr = corr[i];
			index = i / 2;
		}
	}
	
	lag = ((2 * n_samples) - 1 - index) / sampling_rate;
	
	printf("Lag Value is: %lf", lag);

	return 0;
	// END RF_convergence.c
}

// ******************* Support functions ***********************

//=========================== sync_MMSE ======================================================

// Inputs:   numRx, numTx, Ntaps, num_sync_samp, num_rx_samp, sync_symbols (real,imaginary),
//          rx_data (real,imaginary)
// output: the starting time sample index for the comms signal (ipeak = 400)
int sync_MMSE(int numRx, int numTx, int Ntaps, int num_sync_samp, int num_rx_samp, double *sync_symbols_real,
              double *sync_symbols_imag, double *rx_data_sync_real, double *rx_data_sync_imag) {
	double threshold = .7;  // the sync statistic will be between [0,1], .7 is an educated guess
	int delays[Ntaps];      // vector of taps

	for (int k = 0; k < Ntaps; k++) {
		delays[k] = k;
	}

	// Build a weighting vector for the MMSE beamformer. Build it as a decreasing step function where
	// each step corresponds to a new delay tap. This is necessary to give us a sharp, unique peak for our
	// sync statistic.
	double Ntaps_double = (double)Ntaps;  // change Ntaps from int to double
	double sync_weight_taper[numRx * Ntaps];
	for (int k = 0; k < Ntaps; k++) {
		for (int j = 0; j < numRx; j++) {
			sync_weight_taper[k * numRx + j] = (Ntaps_double - k) / Ntaps_double;  // e.g. 1, 4/5, 3/5, 2/5, 1/5
		}
	}

	// Read in the received data
	gsl_matrix_complex *rxSig = NULL;
	rxSig = gsl_matrix_complex_alloc(numRx, num_rx_samp);
	double temp_real, temp_imag;
	gsl_complex temp_complex = gsl_complex_rect(1., 0.);

	for (int i = 0; i < numRx; i++) {
		for (int j = 0; j < num_rx_samp; j++) {
			temp_real = *((rx_data_sync_real + i * num_rx_samp) + j);
			temp_imag = *((rx_data_sync_imag + i * num_rx_samp) + j);
			temp_complex = gsl_complex_rect(temp_real, temp_imag);
			gsl_matrix_complex_set(rxSig, i, j, temp_complex);
		}
	}

	// Read in the sync training sequence
	gsl_matrix_complex *syncSig = NULL;
	syncSig = gsl_matrix_complex_alloc(numTx, num_sync_samp);

	for (int i = 0; i < numTx; i++) {
		for (int j = 0; j < num_sync_samp; j++) {
			temp_real = *((sync_symbols_real + i * num_sync_samp) + j);
			temp_imag = *((sync_symbols_imag + i * num_sync_samp) + j);
			temp_complex = gsl_complex_rect(temp_real, temp_imag);
			gsl_matrix_complex_set(syncSig, i, j, temp_complex);
		}
	}

	// initialize variables
	int sync_flag = 0;  // set to 1 when we have found a possible sync index
	int NsyncStatistics = num_rx_samp - num_sync_samp - 1;
	double sync_statistics[NsyncStatistics];  // to hold our generated sync statistics
	for (int k = 0; k < NsyncStatistics; k++) {
		sync_statistics[k] = 0;
	}

	int index = 0;          // used for counting our sync statistics
	int ipeak = 0;          // this will be our detected sync sample index
	double ipeakVal = 0.0;  // this will be our peak sync statistic value

	double temp_norm_sum = 0.0;
	gsl_complex unity = gsl_complex_rect(1., 0.);         // the complex number 1
	gsl_complex complex_zero = gsl_complex_rect(0., 0.);  // the complex number 0

	// the normilizer should be the norm of the transmitted sync training sequence

	// calculate the norm of the sync training sequence to be used for normalizing our sync statistics
	// printf("temp_norm_sum before loop: %f\r\n\n",temp_norm_sum);
    	KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));
	for (int k = 0; k < num_sync_samp; k++)
	// for(int k=0;k<4;k++)
	{
		//================================================================
		// Debugging...the abs^2 of the syncEstimate grows on each iteration.
		//================================================================

		// printf("temp_real at start of loop: %f\r\n",temp_real);
		temp_complex = gsl_matrix_complex_get(syncSig, 0, k);
		temp_real = gsl_complex_abs2(temp_complex);
		// printf("prior temp_norm_sum inside of loop: %f\r\n\n",temp_norm_sum);
		// printf("temp_real before sum inside of loop: %f\r\n",temp_real);
		temp_norm_sum = temp_norm_sum + temp_real;
		// printf("post temp_norm_sum inside of loop: %f\r\n\n",temp_norm_sum);
	}
    	KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));
	double normalizer;
	normalizer = temp_norm_sum;  // set the normalizer variable
	temp_norm_sum = 0;           // zero out for later use

	// Begin algorithm
	// will hold the relevant section of received data for calculating each test statistic
	gsl_matrix_complex *temp_Rx_data = NULL;
	temp_Rx_data = gsl_matrix_complex_alloc(numRx, num_sync_samp);
	// will hold the space-time delay received data
	gsl_matrix_complex *temp_Rx_space_time_data = NULL;
	temp_Rx_space_time_data = gsl_matrix_complex_alloc(numRx * Ntaps, num_sync_samp);

	//====================
	// initialize all variables to be used in the sync algorithm loops
	int stackedMatSize = numRx * Ntaps;
	gsl_matrix_complex *autoCorrMatrix = NULL;
	autoCorrMatrix = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);  // allocate memory
	gsl_complex sampleToConjugate = gsl_complex_rect(1., 0.);  // To be used to hold the sample to be conjugate in each
	                                                           // loop
	gsl_complex dot_prod = gsl_complex_rect(0., 0.);           // declare type for the dot product accumulator

	gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
	int s;

	gsl_matrix_complex *arrayResponse = NULL;
	arrayResponse = gsl_matrix_complex_alloc(stackedMatSize, 1);  // allocate memory

	gsl_matrix_complex *beamFormer = NULL;
	beamFormer = gsl_matrix_complex_alloc(stackedMatSize, 1);  // allocate memory

	gsl_complex temp_complex_taper = gsl_complex_rect(0., 0.);  // will be used to hold the taper number
	double real_taper = 0;                                      // allocating space
	gsl_complex temp_product = gsl_complex_rect(0., 0.);        // will temporarily hold the tapered beamformer weight

	gsl_matrix_complex *syncEstimate = NULL;
	syncEstimate = gsl_matrix_complex_alloc(numTx, num_sync_samp);

	gsl_matrix_complex *temp_complex_matrix = NULL;
	temp_complex_matrix = gsl_matrix_complex_alloc(1, 1);  // scalar as complex matrix data type

	//             // For debugging.
	//         {
	//             FILE * f = fopen ("Zin.txt", "w");
	//             gsl_matrix_complex_fprintf(f,rxSig,"%f");
	//             fclose(f);
	//         }

	// begin iterating over the sync algorithm
	for (int ll = 0; ll < NsyncStatistics; ll++) {
		index = ll;
		// get the relevant received data for calculating the test statistic for time sample 'index'
		for (int i = 0; i < numRx; i++)  // antenna loop
		{
			for (int j = 0; j < num_sync_samp; j++)  // time samples loop
			{
				temp_complex = gsl_matrix_complex_get(rxSig, i, index + j);  // the index + j-th received sample
				gsl_matrix_complex_set(temp_Rx_data, i, j, temp_complex);
			}
		}

		//         // For debugging
		//         {
		//             FILE * f = fopen ("temp_Rx_data.txt", "w");
		//             gsl_matrix_complex_fprintf(f,temp_Rx_data,"%f");
		//             fclose(f);
		//         }

		// build the space-time delay receive matrix
		for (int i = 0; i < numRx; i++) {              // antenna loop
			for (int j = 0; j < num_sync_samp; j++) {  // time samples loop
				for (int k = 0; k < Ntaps; k++) {      // delay taps loop
					// printf("I: %d, J: %d, K: %d\r\n",i,j,k);
					if (j + k <= num_sync_samp - 1) {
						temp_complex = gsl_matrix_complex_get(temp_Rx_data, i, j + k);  // the k-th delay sample
						gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, temp_complex);
					} else {
						gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, complex_zero);  // set samples
						                                                                                  // at the end
						                                                                                  // to 0+i0
					}
				}
			}
		}

		//         // For debugging
		//         {
		//             FILE * f = fopen ("temp_Rx_space_time_data.txt", "w");
		//             gsl_matrix_complex_fprintf(f,temp_Rx_space_time_data,"%f");
		//             fclose(f);
		//         }

		// build auto-correlation matrix and then invert it
        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp,stackedMatSize));
		for (int i = 0; i < stackedMatSize; i++) {
			for (int k = 0; k < stackedMatSize; k++) {
				dot_prod = gsl_complex_rect(0., 0.);  // reset the dot-product variable to zero
				// dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
				for (int j = 0; j < num_sync_samp; j++) {
					temp_complex = gsl_matrix_complex_get(temp_Rx_space_time_data, i, j);
					sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data, k, j);  // This is the sample to
					                                                                            // conjugate
					sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
					temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate);  // Multiply the two samples
					dot_prod = gsl_complex_add(dot_prod, temp_complex);               // accumulate (dot product)
				}
				// Place this dot product into the cross correlation matrix
				gsl_matrix_complex_set(autoCorrMatrix, i, k, dot_prod);
			}
		}
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp,stackedMatSize));

		//                 //For debugging
		//         {
		//             FILE * f = fopen ("autoCorr.txt", "w");
		//             gsl_matrix_complex_fprintf(f,autoCorrMatrix,"%f");
		//             fclose(f);
		//         }

		// end making the auto-correlation matrix

		// invert auto-correlation matrix

		// Compute the LU decomposition of this matrix

		KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));
		gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
		// Compute the  inverse of the LU decomposition
		gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
		gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
		KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));

		//                     // For debugging
		//         {
		//             FILE * f = fopen ("invAutoCorr.txt", "w");
		//             gsl_matrix_complex_fprintf(f,invAutoCorr,"%f");
		//             fclose(f);
		//         }

		// Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H
		KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_sync_samp));
		gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero,
		               arrayResponse);
		KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_sync_samp));

		//         //For debugging
		//         {
		//             FILE * f = fopen ("arrayResponse.txt", "w");
		//             gsl_matrix_complex_fprintf(f,arrayResponse,"%f");
		//             fclose(f);
		//         }

		// Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
		// This gives us our beamforming vector w

		KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
		KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));

		//                             // For debugging
		//         {
		//             FILE * f = fopen ("beamFormer.txt", "w");
		//             gsl_matrix_complex_fprintf(f,beamFormer,"%f");
		//             fclose(f);
		//         }

		// taper the beamformer weights

		KERN_ENTER(make_label("ZIP[multiply][%d][complex][float64]",stackedMatSize));
		for (int k = 0; k < numRx * Ntaps; k++) {
			// get the beamformer entry
			temp_complex = gsl_matrix_complex_get(beamFormer, k, 0);
			// get the taper entry
			real_taper = sync_weight_taper[k];
			temp_complex_taper = gsl_complex_rect(real_taper, 0.);
			temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
			gsl_matrix_complex_set(beamFormer, k, 0, temp_product);
		}
		KERN_EXIT(make_label("ZIP[multiply][%d][complex][float64]",stackedMatSize));

		//     //For debugging
		//     {
		//         FILE * f = fopen ("beamFormer_tapered.txt", "w");
		//         gsl_matrix_complex_fprintf(f,beamFormer,"%f");
		//         fclose(f);
		//     }

		// Compute the MMSE estimate of what was sent given this data

		KERN_ENTER(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp));
		gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero,
		               syncEstimate);
		KERN_EXIT(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp));

		//     //For debugging
		//     {
		//         FILE * f = fopen ("syncEstimate.txt", "w");
		//         gsl_matrix_complex_fprintf(f,syncEstimate,"%f");
		//         fclose(f);
		//     }

		// calculate the norm of the sync estimate
		temp_norm_sum = 0;

		KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));
		for (int k = 0; k < num_sync_samp; k++) {
			temp_complex = gsl_matrix_complex_get(syncEstimate, 0, k);
			temp_real = gsl_complex_abs2(temp_complex);
			temp_norm_sum = temp_norm_sum + temp_real;
			temp_complex = gsl_complex_rect(0., 0.);
			temp_real = 0;
		}
		KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));
		// normalize and set the sync statistic
		sync_statistics[index] = temp_norm_sum / normalizer;
		temp_norm_sum = 0;  // set to zero for use on next iteration

		// check to see if we crossed the threshold
		if (sync_statistics[index] > threshold) {
			ipeak = index;
			ipeakVal = sync_statistics[index];
			break;
		}

	}  // end of algorithm FOR loop

	// We have crossed the threshold (or never did), now we need to check if the time indices immediately after have a
	// higher sync statistic
	if (ipeak == 0)  // we never synced, set ipeak and ipeakVal to -1 to alert us
	{
		ipeak = -1;
		ipeakVal = -1;
	}

	// Do the same sync algorithm on the next time indices until we see the first decline in value. Then we have our
	// hypothsised sync index
	while (sync_flag == 0 && ipeak != -1) {
		index = index + 1;

		for (int i = 0; i < numRx; i++)  // antenna loop
		{
			for (int j = 0; j < num_sync_samp; j++)  // time samples loop
			{
				temp_complex = gsl_matrix_complex_get(rxSig, i, index + j);  // the index + j-th received sample
				gsl_matrix_complex_set(temp_Rx_data, i, j, temp_complex);
			}
		}

		// build the space-time delay receive matrix
		for (int i = 0; i < numRx; i++) {              // antenna loop
			for (int j = 0; j < num_sync_samp; j++) {  // time samples loop
				for (int k = 0; k < Ntaps; k++) {      // delay taps loop
					// printf("I: %d, J: %d, K: %d\r\n",i,j,k);
					if (j + k <= num_sync_samp - 1) {
						temp_complex = gsl_matrix_complex_get(temp_Rx_data, i, j + k);  // the k-th delay sample
						gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, temp_complex);
					} else {
						gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, complex_zero);  // set samples
						                                                                                  // at the end
						                                                                                  // to 0+i0
					}
				}
			}
		}

		//make the auto-correlation matrix

		KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp,stackedMatSize));
		for (int i = 0; i < stackedMatSize; i++) {
			for (int k = 0; k < stackedMatSize; k++) {
				// dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
				for (int j = 0; j < num_sync_samp; j++) {
					temp_complex = gsl_matrix_complex_get(temp_Rx_space_time_data, i, j);
					sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data, k, j);  // This is the sample to
					                                                                            // conjugate
					sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
					temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate);  // Multiply the two samples
					dot_prod = gsl_complex_add(dot_prod, temp_complex);               // accumulate (dot product)
				}
				// Place this dot product into the cross correlation matrix
				gsl_matrix_complex_set(autoCorrMatrix, i, k, dot_prod);
				dot_prod = gsl_complex_rect(0., 0.);  // reset the dot-product variable to zero
			}
		}
		KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp,stackedMatSize));
		
		// end making the auto-correlation matrix

		// invert auto-correlation matrix
		gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
		// Compute the LU decomposition of this matrix

		KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));
		gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
		// Compute the  inverse of the LU decomposition
		gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
		gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
		KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]",stackedMatSize,stackedMatSize));

		//      gsl_permutation_free(p);

		// Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H

		KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_sync_samp));
		gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero,
		               arrayResponse);
		KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,num_sync_samp));

		// Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
		// This gives us our beamforming vector w

		KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
		KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]",stackedMatSize,stackedMatSize));

		// taper the beamformer weights

		KERN_ENTER(make_label("ZIP[multiply][%d][complex][float64]",stackedMatSize));
		for (int k = 0; k < numRx * Ntaps; k++) {
			// get the beamformer entry
			temp_complex = gsl_matrix_complex_get(beamFormer, k, 0);
			// get the taper entry
			real_taper = sync_weight_taper[k];
			temp_complex_taper = gsl_complex_rect(real_taper, 0.);
			temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
			gsl_matrix_complex_set(beamFormer, k, 0, temp_product);
		}
		KERN_EXIT(make_label("ZIP[multiply][%d][complex][float64]",stackedMatSize));

		// Compute the MMSE estimate of what was sent given this data

		KERN_ENTER(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp));
		gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero,
		               syncEstimate);
		KERN_EXIT(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]",stackedMatSize,num_sync_samp));

		// calculate the norm of the sync estimate

		KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));
		for (int k = 0; k < num_sync_samp; k++) {
			temp_complex = gsl_matrix_complex_get(syncEstimate, 0, k);
			temp_real = gsl_complex_abs2(temp_complex);
			temp_norm_sum = temp_norm_sum + temp_real;
			temp_complex = gsl_complex_rect(0., 0.);
			temp_real = 0;
		}
		KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]",num_sync_samp));

		// normalize and set the sync statistic
		sync_statistics[index] = temp_norm_sum / normalizer;
		temp_norm_sum = 0;  // set to zero for use on next iteration

		// check to see if our sync statistic is increasing, if so, then replace our peak index
		if (sync_statistics[index] > sync_statistics[index - 1]) {
			ipeak = index;
			ipeakVal = sync_statistics[index];
		} else  // our sync statistic decreased, thus we already have our peak index. We are done!
		{
			sync_flag = 1;
			gsl_permutation_free(p);
		}
	}

	//  printf("The final sync index is: %d \n", ipeak);
	//  printf("The final sync peak value is: %f \n", ipeakVal);
	//    printf("%d,%f",ipeak,ipeakVal);

	return ipeak;  // return the index representing the begining of our sync training sequence
}
//======== ENDsync_MMSE support function ======================================================

//================ Matt's LU-factorization approach ================================================

void LU_factorization_projection(gsl_matrix_complex *Z_temp_proj, int numRx, gsl_matrix_complex *S_temp_delay,
                                 int Ntaps_projection, int modulo_N) {
	gsl_complex temp_complex = gsl_complex_rect(1., 0.);

	// calculate the auto-covariance matrix
	gsl_complex unity = gsl_complex_rect(1., 0.);        // the complex number 1
	gsl_complex complexZero = gsl_complex_rect(0., 0.);  // the complex number 0
	gsl_matrix_complex *auto_corr = NULL;
	auto_corr = gsl_matrix_complex_alloc(Ntaps_projection, Ntaps_projection);

	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,modulo_N,Ntaps_projection));
	gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, S_temp_delay, S_temp_delay, complexZero, auto_corr);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,modulo_N,Ntaps_projection));

	gsl_permutation *p = gsl_permutation_alloc(Ntaps_projection);
	int s;

	KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]",Ntaps_projection,Ntaps_projection));	
	gsl_linalg_complex_LU_decomp(auto_corr, p, &s);

	// Compute the  inverse of the LU decomposition
	gsl_matrix_complex *invAutoCorr = NULL;
	invAutoCorr = gsl_matrix_complex_alloc(Ntaps_projection, Ntaps_projection);
	gsl_linalg_complex_LU_invert(auto_corr, p, invAutoCorr);
	KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]",Ntaps_projection,Ntaps_projection));

	gsl_permutation_free(p);

	// Do the orthogonal projection
	gsl_matrix_complex *temp_data = NULL;
	temp_data = gsl_matrix_complex_alloc(Ntaps_projection, modulo_N);
	gsl_matrix_complex *projection = NULL;
	projection = gsl_matrix_complex_alloc(modulo_N, modulo_N);

	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,Ntaps_projection,modulo_N));
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, S_temp_delay, complexZero, temp_data);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,Ntaps_projection,modulo_N));
	
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",modulo_N,Ntaps_projection,modulo_N));
	gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, S_temp_delay, temp_data, complexZero, projection);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",modulo_N,Ntaps_projection,modulo_N));

	// project

	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,modulo_N,modulo_N));
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, Z_temp_proj, projection, complexZero, temp_data);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]",Ntaps_projection,modulo_N,modulo_N));


	// orthogonally project by subtracting the projected data from the original data

	KERN_ENTER(make_label("ZIP[subtract][%d][complex][float64]",numRx*modulo_N));
	for (int i = 0; i < numRx; i++)  // rows of the received data matrix
	{
		for (int j = 0; j < modulo_N; j++) {
			// do the subtraction
			temp_complex =
			    gsl_complex_sub(gsl_matrix_complex_get(Z_temp_proj, i, j), gsl_matrix_complex_get(temp_data, i, j));
			// replace the original received data with the data projected onto subspace orthogonal to S
			gsl_matrix_complex_set(Z_temp_proj, i, j, temp_complex);
		}
	}
	KERN_EXIT(make_label("ZIP[subtract][%d][complex][float64]",numRx*modulo_N));

	// Now we can replace the relevant section of the received data matrix with this newly projected version
}
//================ End LU-factorization approach ================================================

//================ Pulse-Doppler approach ================================================
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

//================ End radar correlator approach ================================================
