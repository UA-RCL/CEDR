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

#include "../include/DashExtras.h"

// #define KERN_ENTER(str);
// #define KERN_EXIT(str);

// char *make_label(const char *fmt, ...);

// ******************* Support functions ***********************

void gsl_fft(double *, double *, size_t);
void gsl_ifft(double *, double *, size_t);

//=========================== sync_MMSE ======================================================

// Inputs:   numRx, numTx, Ntaps, num_sync_samp, num_rx_samp, sync_symbols (real,imaginary),
//          rx_data (real,imaginary)
// output: the starting time sample index for the comms signal (ipeak = 400)
int sync_MMSE(int numRx, int numTx, int Ntaps, int num_sync_samp, int num_rx_samp, double *sync_symbols_real,
              double *sync_symbols_imag, double *rx_data_sync_real, double *rx_data_sync_imag)
{
    double threshold = .7; // the sync statistic will be between [0,1], .7 is an educated guess
    int delays[Ntaps];     // vector of taps

    for (int k = 0; k < Ntaps; k++)
    {
        delays[k] = k;
    }

    // Build a weighting vector for the MMSE beamformer. Build it as a decreasing step function where
    // each step corresponds to a new delay tap. This is necessary to give us a sharp, unique peak for our
    // sync statistic.
    double Ntaps_double = (double)Ntaps; // change Ntaps from int to double
    double sync_weight_taper[numRx * Ntaps];
    for (int k = 0; k < Ntaps; k++)
    {
        for (int j = 0; j < numRx; j++)
        {
            sync_weight_taper[k * numRx + j] = (Ntaps_double - k) / Ntaps_double; // e.g. 1, 4/5, 3/5, 2/5, 1/5
        }
    }

    // Read in the received data
    gsl_matrix_complex *rxSig = NULL;
    rxSig = gsl_matrix_complex_alloc(numRx, num_rx_samp);
    double temp_real, temp_imag;
    gsl_complex temp_complex = gsl_complex_rect(1., 0.);

    for (int i = 0; i < numRx; i++)
    {
        for (int j = 0; j < num_rx_samp; j++)
        {
            temp_real = *((rx_data_sync_real + i * num_rx_samp) + j);
            temp_imag = *((rx_data_sync_imag + i * num_rx_samp) + j);
            temp_complex = gsl_complex_rect(temp_real, temp_imag);
            gsl_matrix_complex_set(rxSig, i, j, temp_complex);
        }
    }

    // Read in the sync training sequence
    gsl_matrix_complex *syncSig = NULL;
    syncSig = gsl_matrix_complex_alloc(numTx, num_sync_samp);

    for (int i = 0; i < numTx; i++)
    {
        for (int j = 0; j < num_sync_samp; j++)
        {
            temp_real = *((sync_symbols_real + i * num_sync_samp) + j);
            temp_imag = *((sync_symbols_imag + i * num_sync_samp) + j);
            temp_complex = gsl_complex_rect(temp_real, temp_imag);
            gsl_matrix_complex_set(syncSig, i, j, temp_complex);
        }
    }

    // initialize variables
    int sync_flag = 0; // set to 1 when we have found a possible sync index
    int NsyncStatistics = num_rx_samp - num_sync_samp - 1;
    double sync_statistics[NsyncStatistics]; // to hold our generated sync statistics
    for (int k = 0; k < NsyncStatistics; k++)
    {
        sync_statistics[k] = 0;
    }

    int index = 0;         // used for counting our sync statistics
    int ipeak = 0;         // this will be our detected sync sample index
    double ipeakVal = 0.0; // this will be our peak sync statistic value

    double temp_norm_sum = 0.0;
    gsl_complex unity = gsl_complex_rect(1., 0.);        // the complex number 1
    gsl_complex complex_zero = gsl_complex_rect(0., 0.); // the complex number 0

    // the normilizer should be the norm of the transmitted sync training sequence

    // calculate the norm of the sync training sequence to be used for normalizing our sync statistics
    // printf("temp_norm_sum before loop: %f\r\n\n",temp_norm_sum);
    KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));
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
    KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));
    double normalizer;
    normalizer = temp_norm_sum; // set the normalizer variable
    temp_norm_sum = 0;          // zero out for later use

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
    autoCorrMatrix = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize); // allocate memory
    gsl_complex sampleToConjugate = gsl_complex_rect(1., 0.);                  // To be used to hold the sample to be conjugate in each
                                                                               // loop
    gsl_complex dot_prod = gsl_complex_rect(0., 0.);                           // declare type for the dot product accumulator

    gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
    int s;

    gsl_matrix_complex *arrayResponse = NULL;
    arrayResponse = gsl_matrix_complex_alloc(stackedMatSize, 1); // allocate memory

    gsl_matrix_complex *beamFormer = NULL;
    beamFormer = gsl_matrix_complex_alloc(stackedMatSize, 1); // allocate memory

    gsl_complex temp_complex_taper = gsl_complex_rect(0., 0.); // will be used to hold the taper number
    double real_taper = 0;                                     // allocating space
    gsl_complex temp_product = gsl_complex_rect(0., 0.);       // will temporarily hold the tapered beamformer weight

    gsl_matrix_complex *syncEstimate = NULL;
    syncEstimate = gsl_matrix_complex_alloc(numTx, num_sync_samp);

    gsl_matrix_complex *temp_complex_matrix = NULL;
    temp_complex_matrix = gsl_matrix_complex_alloc(1, 1); // scalar as complex matrix data type

    //             // For debugging.
    //         {
    //             FILE * f = fopen ("Zin.txt", "w");
    //             gsl_matrix_complex_fprintf(f,rxSig,"%f");
    //             fclose(f);
    //         }

    // begin iterating over the sync algorithm
    for (int ll = 0; ll < NsyncStatistics; ll++)
    {
        index = ll;
        // get the relevant received data for calculating the test statistic for time sample 'index'
        for (int i = 0; i < numRx; i++) // antenna loop
        {
            for (int j = 0; j < num_sync_samp; j++) // time samples loop
            {
                temp_complex = gsl_matrix_complex_get(rxSig, i, index + j); // the index + j-th received sample
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
        for (int i = 0; i < numRx; i++)
        { // antenna loop
            for (int j = 0; j < num_sync_samp; j++)
            { // time samples loop
                for (int k = 0; k < Ntaps; k++)
                { // delay taps loop
                    // printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                    if (j + k <= num_sync_samp - 1)
                    {
                        temp_complex = gsl_matrix_complex_get(temp_Rx_data, i, j + k); // the k-th delay sample
                        gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, temp_complex);
                    }
                    else
                    {
                        gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, complex_zero); // set samples
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
        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp, stackedMatSize));
        for (int i = 0; i < stackedMatSize; i++)
        {
            for (int k = 0; k < stackedMatSize; k++)
            {
                dot_prod = gsl_complex_rect(0., 0.); // reset the dot-product variable to zero
                // dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
                for (int j = 0; j < num_sync_samp; j++)
                {
                    temp_complex = gsl_matrix_complex_get(temp_Rx_space_time_data, i, j);
                    sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data, k, j); // This is the sample to
                                                                                               // conjugate
                    sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
                    temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate); // Multiply the two samples
                    dot_prod = gsl_complex_add(dot_prod, temp_complex);              // accumulate (dot product)
                }
                // Place this dot product into the cross correlation matrix
                gsl_matrix_complex_set(autoCorrMatrix, i, k, dot_prod);
            }
        }
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp, stackedMatSize));

        //                 //For debugging
        //         {
        //             FILE * f = fopen ("autoCorr.txt", "w");
        //             gsl_matrix_complex_fprintf(f,autoCorrMatrix,"%f");
        //             fclose(f);
        //         }

        // end making the auto-correlation matrix

        // invert auto-correlation matrix

        // Compute the LU decomposition of this matrix

        KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]", stackedMatSize, stackedMatSize));
        gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
        // Compute the  inverse of the LU decomposition
        gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
        gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
        KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]", stackedMatSize, stackedMatSize));

        //                     // For debugging
        //         {
        //             FILE * f = fopen ("invAutoCorr.txt", "w");
        //             gsl_matrix_complex_fprintf(f,invAutoCorr,"%f");
        //             fclose(f);
        //         }

        // Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H
        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, num_sync_samp));
        gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero,
                       arrayResponse);
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, num_sync_samp));

        //         //For debugging
        //         {
        //             FILE * f = fopen ("arrayResponse.txt", "w");
        //             gsl_matrix_complex_fprintf(f,arrayResponse,"%f");
        //             fclose(f);
        //         }

        // Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
        // This gives us our beamforming vector w

        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, stackedMatSize));
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, stackedMatSize));

        //                             // For debugging
        //         {
        //             FILE * f = fopen ("beamFormer.txt", "w");
        //             gsl_matrix_complex_fprintf(f,beamFormer,"%f");
        //             fclose(f);
        //         }

        // taper the beamformer weights

        KERN_ENTER(make_label("ZIP[multiply][%d][complex][float64]", stackedMatSize));
        for (int k = 0; k < numRx * Ntaps; k++)
        {
            // get the beamformer entry
            temp_complex = gsl_matrix_complex_get(beamFormer, k, 0);
            // get the taper entry
            real_taper = sync_weight_taper[k];
            temp_complex_taper = gsl_complex_rect(real_taper, 0.);
            temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
            gsl_matrix_complex_set(beamFormer, k, 0, temp_product);
        }
        KERN_EXIT(make_label("ZIP[multiply][%d][complex][float64]", stackedMatSize));

        //     //For debugging
        //     {
        //         FILE * f = fopen ("beamFormer_tapered.txt", "w");
        //         gsl_matrix_complex_fprintf(f,beamFormer,"%f");
        //         fclose(f);
        //     }

        // Compute the MMSE estimate of what was sent given this data

        KERN_ENTER(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp));
        gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero,
                       syncEstimate);
        KERN_EXIT(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp));

        //     //For debugging
        //     {
        //         FILE * f = fopen ("syncEstimate.txt", "w");
        //         gsl_matrix_complex_fprintf(f,syncEstimate,"%f");
        //         fclose(f);
        //     }

        // calculate the norm of the sync estimate
        temp_norm_sum = 0;

        KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));
        for (int k = 0; k < num_sync_samp; k++)
        {
            temp_complex = gsl_matrix_complex_get(syncEstimate, 0, k);
            temp_real = gsl_complex_abs2(temp_complex);
            temp_norm_sum = temp_norm_sum + temp_real;
            temp_complex = gsl_complex_rect(0., 0.);
            temp_real = 0;
        }
        KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));
        // normalize and set the sync statistic
        sync_statistics[index] = temp_norm_sum / normalizer;
        temp_norm_sum = 0; // set to zero for use on next iteration

        // check to see if we crossed the threshold
        if (sync_statistics[index] > threshold)
        {
            ipeak = index;
            ipeakVal = sync_statistics[index];
            break;
        }

    } // end of algorithm FOR loop

    // We have crossed the threshold (or never did), now we need to check if the time indices immediately after have a
    // higher sync statistic
    if (ipeak == 0) // we never synced, set ipeak and ipeakVal to -1 to alert us
    {
        ipeak = -1;
        ipeakVal = -1;
    }

    // Do the same sync algorithm on the next time indices until we see the first decline in value. Then we have our
    // hypothsised sync index
    while (sync_flag == 0 && ipeak != -1)
    {
        index = index + 1;

        for (int i = 0; i < numRx; i++) // antenna loop
        {
            for (int j = 0; j < num_sync_samp; j++) // time samples loop
            {
                temp_complex = gsl_matrix_complex_get(rxSig, i, index + j); // the index + j-th received sample
                gsl_matrix_complex_set(temp_Rx_data, i, j, temp_complex);
            }
        }

        // build the space-time delay receive matrix
        for (int i = 0; i < numRx; i++)
        { // antenna loop
            for (int j = 0; j < num_sync_samp; j++)
            { // time samples loop
                for (int k = 0; k < Ntaps; k++)
                { // delay taps loop
                    // printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                    if (j + k <= num_sync_samp - 1)
                    {
                        temp_complex = gsl_matrix_complex_get(temp_Rx_data, i, j + k); // the k-th delay sample
                        gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, temp_complex);
                    }
                    else
                    {
                        gsl_matrix_complex_set(temp_Rx_space_time_data, numRx * k + i, j, complex_zero); // set samples
                                                                                                         // at the end
                                                                                                         // to 0+i0
                    }
                }
            }
        }

        //make the auto-correlation matrix

        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp, stackedMatSize));
        for (int i = 0; i < stackedMatSize; i++)
        {
            for (int k = 0; k < stackedMatSize; k++)
            {
                // dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
                for (int j = 0; j < num_sync_samp; j++)
                {
                    temp_complex = gsl_matrix_complex_get(temp_Rx_space_time_data, i, j);
                    sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data, k, j); // This is the sample to
                                                                                               // conjugate
                    sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
                    temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate); // Multiply the two samples
                    dot_prod = gsl_complex_add(dot_prod, temp_complex);              // accumulate (dot product)
                }
                // Place this dot product into the cross correlation matrix
                gsl_matrix_complex_set(autoCorrMatrix, i, k, dot_prod);
                dot_prod = gsl_complex_rect(0., 0.); // reset the dot-product variable to zero
            }
        }
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp, stackedMatSize));

        // end making the auto-correlation matrix

        // invert auto-correlation matrix
        gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
        // Compute the LU decomposition of this matrix

        KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]", stackedMatSize, stackedMatSize));
        gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
        // Compute the  inverse of the LU decomposition
        gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
        gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
        KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]", stackedMatSize, stackedMatSize));

        //      gsl_permutation_free(p);

        // Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H

        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, num_sync_samp));
        gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero,
                       arrayResponse);
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, num_sync_samp));

        // Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
        // This gives us our beamforming vector w

        KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, stackedMatSize));
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
        KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-1][complex][float64]", stackedMatSize, stackedMatSize));

        // taper the beamformer weights

        KERN_ENTER(make_label("ZIP[multiply][%d][complex][float64]", stackedMatSize));
        for (int k = 0; k < numRx * Ntaps; k++)
        {
            // get the beamformer entry
            temp_complex = gsl_matrix_complex_get(beamFormer, k, 0);
            // get the taper entry
            real_taper = sync_weight_taper[k];
            temp_complex_taper = gsl_complex_rect(real_taper, 0.);
            temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
            gsl_matrix_complex_set(beamFormer, k, 0, temp_product);
        }
        KERN_EXIT(make_label("ZIP[multiply][%d][complex][float64]", stackedMatSize));

        // Compute the MMSE estimate of what was sent given this data

        KERN_ENTER(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp));
        gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero,
                       syncEstimate);
        KERN_EXIT(make_label("GEMM[Ar-1][Ac-%d][Bc-%d][complex][float64]", stackedMatSize, num_sync_samp));

        // calculate the norm of the sync estimate

        KERN_ENTER(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));
        for (int k = 0; k < num_sync_samp; k++)
        {
            temp_complex = gsl_matrix_complex_get(syncEstimate, 0, k);
            temp_real = gsl_complex_abs2(temp_complex);
            temp_norm_sum = temp_norm_sum + temp_real;
            temp_complex = gsl_complex_rect(0., 0.);
            temp_real = 0;
        }
        KERN_EXIT(make_label("GEMV[Ar-1][Ac-%d][complex][float64]", num_sync_samp));

        // normalize and set the sync statistic
        sync_statistics[index] = temp_norm_sum / normalizer;
        temp_norm_sum = 0; // set to zero for use on next iteration

        // check to see if our sync statistic is increasing, if so, then replace our peak index
        if (sync_statistics[index] > sync_statistics[index - 1])
        {
            ipeak = index;
            ipeakVal = sync_statistics[index];
        }
        else // our sync statistic decreased, thus we already have our peak index. We are done!
        {
            sync_flag = 1;
            gsl_permutation_free(p);
        }
    }

    //  printf("The final sync index is: %d \n", ipeak);
    //  printf("The final sync peak value is: %f \n", ipeakVal);
    //    printf("%d,%f",ipeak,ipeakVal);

    return ipeak; // return the index representing the begining of our sync training sequence
}
//======== ENDsync_MMSE support function ======================================================

//================ Matt's LU-factorization approach ================================================

//void LU_factorization_projection(gsl_matrix_complex *Z_temp_proj, int numRx, gsl_matrix_complex *S_temp_delay,
//                                 int Ntaps_projection, int modulo_N)
//{
//    gsl_complex temp_complex = gsl_complex_rect(1., 0.);
//
//    // calculate the auto-covariance matrix
//    gsl_complex unity = gsl_complex_rect(1., 0.);       // the complex number 1
//    gsl_complex complexZero = gsl_complex_rect(0., 0.); // the complex number 0
//    gsl_matrix_complex *auto_corr = NULL;
//    auto_corr = gsl_matrix_complex_alloc(Ntaps_projection, Ntaps_projection);
//
//    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, modulo_N, Ntaps_projection));
//    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, S_temp_delay, S_temp_delay, complexZero, auto_corr);
//    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, modulo_N, Ntaps_projection));
//
//    gsl_permutation *p = gsl_permutation_alloc(Ntaps_projection);
//    int s;
//
//    KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d]", Ntaps_projection, Ntaps_projection));
//    gsl_linalg_complex_LU_decomp(auto_corr, p, &s);
//
//    // Compute the  inverse of the LU decomposition
//    gsl_matrix_complex *invAutoCorr = NULL;
//    invAutoCorr = gsl_matrix_complex_alloc(Ntaps_projection, Ntaps_projection);
//    gsl_linalg_complex_LU_invert(auto_corr, p, invAutoCorr);
//    KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d]", Ntaps_projection, Ntaps_projection));
//
//    gsl_permutation_free(p);
//
//    // Do the orthogonal projection
//    gsl_matrix_complex *temp_data = NULL;
//    temp_data = gsl_matrix_complex_alloc(Ntaps_projection, modulo_N);
//    gsl_matrix_complex *projection = NULL;
//    projection = gsl_matrix_complex_alloc(modulo_N, modulo_N);
//
//    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, Ntaps_projection, modulo_N));
//    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, S_temp_delay, complexZero, temp_data);
//    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, Ntaps_projection, modulo_N));
//
//    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", modulo_N, Ntaps_projection, modulo_N));
//    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, S_temp_delay, temp_data, complexZero, projection);
//    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", modulo_N, Ntaps_projection, modulo_N));
//
//    // project
//
//    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, modulo_N, modulo_N));
//    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, Z_temp_proj, projection, complexZero, temp_data);
//    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float64]", Ntaps_projection, modulo_N, modulo_N));
//
//    // orthogonally project by subtracting the projected data from the original data
//
//    KERN_ENTER(make_label("ZIP[subtract][%d][complex][float64]", numRx * modulo_N));
//    for (int i = 0; i < numRx; i++) // rows of the received data matrix
//    {
//        for (int j = 0; j < modulo_N; j++)
//        {
//            // do the subtraction
//            temp_complex =
//                gsl_complex_sub(gsl_matrix_complex_get(Z_temp_proj, i, j), gsl_matrix_complex_get(temp_data, i, j));
//            // replace the original received data with the data projected onto subspace orthogonal to S
//            gsl_matrix_complex_set(Z_temp_proj, i, j, temp_complex);
//        }
//    }
//    KERN_EXIT(make_label("ZIP[subtract][%d][complex][float64]", numRx * modulo_N));
//
//    // Now we can replace the relevant section of the received data matrix with this newly projected version
//}
//================ End LU-factorization approach ================================================

//================ Pulse-Doppler approach ================================================
//void __attribute__((always_inline)) xcorr(double *x, double *y, size_t n_samp, double *corr)
//{
//    size_t len = 2 * n_samp - 1;
//
//    double *c = malloc(2 * len * sizeof(double));
//    double *d = malloc(2 * len * sizeof(double));
//
//    size_t x_count = 0;
//    size_t y_count = 0;
//
//    for (size_t i = 0; i < 2 * len; i += 2)
//    {
//        if (i / 2 > n_samp - 1)
//        {
//            c[i] = x[x_count];
//            c[i + 1] = x[x_count + 1];
//            x_count += 2;
//        }
//        else
//        {
//            c[i] = 0;
//            c[i + 1] = 0;
//        }
//
//        if (i > n_samp)
//        {
//            d[i] = 0;
//            d[i + 1] = 0;
//        }
//        else
//        {
//            d[i] = y[y_count];
//            d[i + 1] = y[y_count + 1];
//            y_count += 2;
//        }
//    }
//
//    double *X1 = malloc(2 * len * sizeof(double));
//    double *X2 = malloc(2 * len * sizeof(double));
//    double *corr_freq = malloc(2 * len * sizeof(double));
//    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", len));
//    gsl_fft(c, X1, len);
//    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", len));
//    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", len));
//    gsl_fft(d, X2, len);
//    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", len));
//
//    KERN_ENTER(make_label("ZIP[multiply][complex][float64][%d]", len));
//    for (size_t i = 0; i < 2 * len; i += 2)
//    {
//        corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
//        corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
//    }
//    KERN_EXIT(make_label("ZIP[multiply][complex][float64][%d]", len));
//
//    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", len));
//    gsl_ifft(corr_freq, corr, len);
//    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", len));
//}

//================ End radar correlator approach ================================================