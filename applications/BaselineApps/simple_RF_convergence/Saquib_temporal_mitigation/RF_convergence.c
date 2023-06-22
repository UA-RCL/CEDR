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

#include <stdio.h>
#include <stdlib.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
//#include<iostream>
//#include<fstream>
#include<math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

// #define PROGPATH DASH_DATA "Dash-RadioCorpus/RF_convergence/"
// #define RX PROGPATH "rx.dat"
// #define DATAEST PROGPATH "dataEstimate.txt" 
// #define XTRAIN PROGPATH "x_Training.dat"
// #define SYNCTRAIN PROGPATH "sync_symbols.dat"

//For saquib's temporal mitigation code
// #include "inverse.h"

//#define PROGPATH DASH_DATA "Dash-RadioCorpus/temporal_mitigation/"
//#define ZIN PROGPATH "z.txt"
//#define ZIMAGIN PROGPATH "zimag.txt"
//#define SIN PROGPATH "s.txt"
//#define SIMAGIN PROGPATH "simag.txt"

#define N 4
#define M 64



void getCofactor(float A[N*N], float temp[N*N], int p, int q, int n);
//#pragma SDS data data_mover(m:AXIDMA_SIMPLE,invOut:AXIDMA_SIMPLE)

//#pragma SDS data access_pattern(m:SEQUENTIAL,invOut:SEQUENTIAL)
void alternateinverse( float m[16], float invOut[16]);
//#pragma SDS data data_mover(m:AXIDMA_SIMPLE,invOut:AXIDMA_SIMPLE)
void scalableinverse(float m[16], float invOut[16]);
float determinant(float A[N*N], int n);
void adjoint(float A[N*N],float adj[N][N]);
void inverse(float A[N*N], float inverse[N*N]);
void display(float *A,int ,int);
void realpart(float A[N*N], float B[N*N] , float inv1[N*N] ,float inv2[N*N] , float intmedt1[N*N]);
void imagpart(float A[N*N],float B[N*N],float inv1[N*N], float inv2[N*N],float intmedt2[N*N]);
//#pragma SDS data access_pattern(A:SEQUENTIAL,B:SEQUENTIAL,C:SEQUENTIAL)
void mmultiply(float A[N*N],float B[N*N],float C[N*N]);
//#pragma SDS data access_pattern(S:SEQUENTIAL ,Si:SEQUENTIAL,Shermitian:SEQUENTIAL , Shermitianimag:SEQUENTIAL)
void hermitian(float S[N*M] ,float Si[N*M],float Shermitian[N*M],float Shermitianimag[N*M]);
float divide(float,float);


//#pragma SDS data access_pattern(A:SEQUENTIAL, Ai:SEQUENTIAL ,B:SEQUENTIAL,Bi:SEQUENTIAL, C:SEQUENTIAL,Ci:SEQUENTIAL)
void mmult (float A[N*N],float Ai[N*N],float B[N*N],float Bi[N*N], float C[N*N],float Ci[N*N]);

//#pragma SDS data access_pattern(A:SEQUENTIAL, Ai:SEQUENTIAL ,B:SEQUENTIAL,Bi:SEQUENTIAL, C:SEQUENTIAL,Ci:SEQUENTIAL)
void mmult4 (float A[N*N],float Ai[N*N],float B[N*N],float Bi[N*N], float C[N*N],float Ci[N*N]);


//#pragma SDS data access_pattern(A:SEQUENTIAL, Ai:SEQUENTIAL ,B:SEQUENTIAL,Bi:SEQUENTIAL, C:SEQUENTIAL,Ci:SEQUENTIAL)
void mmult64 (float A[N*N],float Ai[N*N], float B[N*M], float Bi[N*M] ,float C[N*M], float Ci[N*M]);


//#pragma SDS data access_pattern(A:SEQUENTIAL, Ai:SEQUENTIAL ,B:SEQUENTIAL,Bi:SEQUENTIAL, C:SEQUENTIAL,Ci:SEQUENTIAL)
void msub(float A[N*M],float Ai[N*M], float B[N*M], float Bi[N*M] ,float C[N*M], float Ci[N*M]);

// end Saquib's


//Prototypes of the functions to be called by main()

//returns the starting time sample index for the comms signal (ipeak = 400)
int sync_MMSE(int numRx, int numTx, int Ntaps, int num_sync_samp, int num_rx_samp, double *sync_symbols_real, double *sync_symbols_imag, double *rx_data_sync_real, double *rx_data_sync_imag);
void temporal_mitigation(float *Z, float *Zi, float *S, float *Si, float *zres, float *zresimag);
void QR_factorization_projection(float *temp_Z_real, float *temp_Z_imag, float *temp_S_real, float *temp_S_imag, float *zres_real, float *zres_imag);

int main(){
    // Read in data structures and parameters
    int numRx = 4;  //ALWAYS KEEP AS 4!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation code)
    int numTx = 1;  //ALWAYS KEEP AS 1!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation code)
    int Ntaps = 5;  //for sync and MMSE beamforming
    int initial_offset = -1; //begin at the -1th delay tap for the STAP data matrix
    int Ntaps_projection = 4; //ALWAYS KEEP AS 4!!!!!!!!!!!!!! (to be compatible with Saquib's temporal mitigation code)
    //error checking
    if(numRx != Ntaps_projection)
    {
        printf("Number of receive antennas needs to be equal to Ntaps for the projection matrix (both equal to 4).\r\n");
        printf("numRx: %d, Ntaps_projection: %d\r\n",numRx, Ntaps_projection);
        exit(4);
    }
    int modulo_N = 64; //We need to call Saquib's temporal mitigation code in blocks of 64 time samples
    int n_sync_blocks = 4; //number of comms sync blocks of 64 time samples
    int n_trn_blocks = 4; //number of comms training blocks of 64 time samples
    int n_data_blocks = 280; //number of comms data blocks of 64 time samples
    int n_start_zeros = 400;
    int num_sync_samp = n_sync_blocks*modulo_N;
    int n_sync_trn_zeros = 100; //zero padding between comms sync and training signals
    int num_trn_samp = n_trn_blocks*modulo_N;
    int num_trn_data_zeros = 100; //zero padding between comms training and data signals
    int num_data_samp = n_data_blocks*modulo_N;
    int n_data_end_zeros = 18; //zero padding at the end of the comms signal. This number needs to be hard coded so that we have total time samples of 19050
    int num_rx_samp = 19050;
    //error check
    int samp_sum = n_start_zeros+num_sync_samp+n_sync_trn_zeros+num_trn_samp+num_trn_data_zeros+num_data_samp+n_data_end_zeros;
    if(samp_sum != num_rx_samp)
    {
        printf("Number of total Rx time samples is not correct?\r\n");
        printf("Sum of samples: %d, N Rx samples: %d\r\n",samp_sum, num_rx_samp);
        exit(4);
    }
    //QAM comms constellation parameters (hard coded for now)
    int N_constellation_points = 4;
    gsl_matrix_complex *QAM_constellation = NULL;
    QAM_constellation = gsl_matrix_complex_alloc(1,N_constellation_points);
    gsl_matrix_complex_set(QAM_constellation,0,0,gsl_complex_rect(1.,0.));
    gsl_matrix_complex_set(QAM_constellation,0,1,gsl_complex_rect(0.,1.));
    gsl_matrix_complex_set(QAM_constellation,0,2,gsl_complex_rect(-1.,0.));
    gsl_matrix_complex_set(QAM_constellation,0,3,gsl_complex_rect(0.,-1.));
    
    /*
     * Read in raw data - and form into signal matrix 
     */
    gsl_matrix_complex *rxSig = NULL;
    FILE *rawDataFile = fopen("rx.dat","r");
    //FILE *rawDataFile = fopen(RX,"r");
    rxSig = gsl_matrix_complex_alloc(numRx,num_rx_samp);
    double temp_real,temp_imag;
    gsl_complex temp_complex = gsl_complex_rect(1.,0.);
    int dataNumPulses;
    fread(&dataNumPulses,sizeof(int),1,rawDataFile);
    if((numRx*num_rx_samp) != dataNumPulses) {
        printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
        printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, numRx*num_rx_samp);
        exit(4);
    }
    for(int i = 0; i < num_rx_samp; i++) {
        for(int j = 0; j < numRx; j++){
            fread(&temp_real, sizeof(double), 1, rawDataFile);
            fread(&temp_imag, sizeof(double), 1, rawDataFile);
            temp_complex = gsl_complex_rect(temp_real,temp_imag);
            gsl_matrix_complex_set(rxSig,j,i,temp_complex);
        }
    }
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
    FILE *rawDataFile2 = fopen("sync_symbols.dat","r");
    //FILE *rawDataFile2 = fopen(SYNCTRAIN,"r");
    sync_symbols = gsl_matrix_complex_alloc(numTx,num_sync_samp);
    fread(&dataNumPulses,sizeof(int),1,rawDataFile2);
    if((num_sync_samp) != dataNumPulses) {
        printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
        printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, num_sync_samp);
        exit(4);
    }
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < num_sync_samp; j++){
            fread(&temp_real, sizeof(double), 1, rawDataFile2);
            fread(&temp_imag, sizeof(double), 1, rawDataFile2);
            temp_complex = gsl_complex_rect(temp_real,temp_imag);
            gsl_matrix_complex_set(sync_symbols,i,j,temp_complex);
        }
    }
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
    FILE *rawDataFile3 = fopen("x_Training.dat","r");
    //FILE *rawDataFile3 = fopen(XTRAIN,"r");
    x_Training = gsl_matrix_complex_alloc(1,num_trn_samp);
    fread(&dataNumPulses,sizeof(int),1,rawDataFile3);
    if((num_trn_samp) != dataNumPulses) {
        printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
        printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, num_trn_samp);
        exit(4);
    }
    for(int i = 0; i < 1; i++) 
    {    
        for(int j = 0; j < num_trn_samp; j++)
        {
            fread(&temp_real, sizeof(double), 1, rawDataFile2);
            fread(&temp_imag, sizeof(double), 1, rawDataFile2);
            temp_complex = gsl_complex_rect(temp_real,temp_imag);
            gsl_matrix_complex_set(x_Training,i,j,temp_complex);
        }
    }
    fclose(rawDataFile2);
    printf("Done reading in training reference.\r\n");

//     //debugging: check to see if we read the data in correctly (done)
//     {
//         FILE * f = fopen ("readInX_training.txt", "w");
//         gsl_matrix_complex_fprintf(f,x_Training,"%f");
//         fclose(f);
//     }
    
// =================== begin synchronization to the comms signal ======================================
    //Get real and imaginary parts of the complex matrices (both sync and rx data)
   //Dynamic memory initialization
    double *sync_symbols_real,*sync_symbols_imag;
    double *rx_data_sync_real,*rx_data_sync_imag;
    //Dynamic memory allocation
     sync_symbols_real = (double *)malloc(numTx*num_sync_samp*sizeof(double));
    sync_symbols_imag = (double *)malloc(numTx*num_sync_samp*sizeof(double));
    
    rx_data_sync_real = (double *)malloc(numRx*num_rx_samp*sizeof(double));
    rx_data_sync_imag = (double *)malloc(numRx*num_rx_samp*sizeof(double));
     
//     double sync_symbols_imag[numTx,num_sync_samp];
//     double rx_data_real[numRx,num_rx_samp];
//     double rx_data_imag[numRx,num_rx_samp];
  
    for(int i = 0; i < numTx; i++)
    {
        for(int j = 0; j < num_sync_samp; j++)
        {
            //QUESTION: Does this dynamic memory access automatically account for the size of the data type?
            sync_symbols_real[i*num_sync_samp + j] = GSL_REAL(gsl_matrix_complex_get(sync_symbols,i,j));
           // *((sync_symbols_real+ i*num_sync_samp)+j) = GSL_REAL(gsl_matrix_complex_get(&sync_symbols,i,j));
            sync_symbols_imag[i*num_sync_samp + j] = GSL_IMAG(gsl_matrix_complex_get(sync_symbols,i,j));
          //  *((sync_symbols_imag+ i*num_sync_samp)+j) = GSL_IMAG(gsl_matrix_complex_get(&sync_symbols,i,j));
//             fread(&temp_real, sizeof(double), 1, rawDataFile);
//             fread(&temp_imag, sizeof(double), 1, rawDataFile);
//            temp_real = *( (rx_data_real+ i*num_rx_samp) + j);
//            temp_imag = *( (rx_data_imag+ i*num_rx_samp) + j);
//      temp_complex = gsl_complex_rect(temp_real,temp_imag);
//        gsl_matrix_complex_set(rxSig,i,j,temp_complex);
        }
     }
    
    for(int i = 0; i < numRx; i++)
    {    
       for(int j = 0; j < num_rx_samp; j++)
       {
           rx_data_sync_real[i*num_rx_samp + j] = GSL_REAL(gsl_matrix_complex_get(rxSig,i,j));
          // *((rx_data_real+ i*num_rx_samp)+j) = GSL_REAL(gsl_matrix_complex_get(rxSig,i,j));
           rx_data_sync_imag[i*num_rx_samp + j] = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,j));
          // *((rx_data_imag+ i*num_rx_samp)+j) = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,j));
//             fread(&temp_real, sizeof(double), 1, rawDataFile);
//             fread(&temp_imag, sizeof(double), 1, rawDataFile);
//            temp_real = *( (rx_data_real+ i*num_rx_samp) + j);
//            temp_imag = *( (rx_data_imag+ i*num_rx_samp) + j);
//      temp_complex = gsl_complex_rect(temp_real,temp_imag);
//        gsl_matrix_complex_set(rxSig,i,j,temp_complex);
        }
    }
    
    
    
    
    
//     for(int i=0;i<numRx;i++)  //received data
//     {
//         for(int j=0; j< num_rx_samp; j++)
//         {
//             rx_data_real[i,j] = GSL_REAL(gsl_matrix_complex_get(rxSig,i,j));
//             rx_data_imag[i,j] = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,j));
//         }
//     }
//     
//     for(int i=0;i<numTx;i++) //sync symbols
//     {
//         for(int j=0; j< num_sync_samp; j++)
//         {
//             sync_symbols_real[i,j] = GSL_REAL(gsl_matrix_complex_get(sync_symbols,i,j));
//             sync_symbols_imag[i,j] = GSL_IMAG(gsl_matrix_complex_get(sync_symbols,i,j));
//         }
//     }

  //get the time index at which the comms signal begins
  int ipeak = sync_MMSE(numRx, numTx, Ntaps, num_sync_samp, num_rx_samp, sync_symbols_real, sync_symbols_imag, rx_data_sync_real, rx_data_sync_imag);
    
  // =================== end sync to the comms signal ======================================  
    
  // =================== begin MMSE comms signal estimation ================================
  
  //beginning at the first comms training sample we build the Space-time receive matrix
  int comms_sync_start_idx = ipeak;
  int comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros; 
  int comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;
  
  // Make the delayed matrix, rxSigDelays (-1 + LOS + 3 delay taps, say)
 //   int numDelayTaps = 4;
    int stackedMatSize = Ntaps * numRx;
    double zero = 0;
    gsl_complex complexZero = gsl_complex_rect(zero, zero);
    // Slide rows of rxSig matrix to the left by 1 sample. Fill in with zeros at the end
    gsl_matrix_complex *rxSigDelays = NULL;
    rxSigDelays = gsl_matrix_complex_alloc(stackedMatSize,num_trn_samp); //allocate memory for the space-time matrix
        //fill in the space-time matrix using rxSig. Zero pad at the end.
    for(int i = 0; i < numRx; i++) { //antenna loop
        for(int j = 0; j < num_trn_samp; j++){ //time samples loop
            //place the samples for antenna i for LOS and all delays
            for(int k = initial_offset; k < Ntaps + initial_offset; k++){ //delay taps loop
                //printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                if (j+k <= num_trn_samp - 1 + initial_offset){
                    temp_complex = gsl_matrix_complex_get(rxSig,i,comms_trn_start_idx+j+k); //the k-th delay sample
                    gsl_matrix_complex_set(rxSigDelays,numRx*(k-initial_offset)+i,j,temp_complex);
                }
                else {
                    gsl_matrix_complex_set(rxSigDelays,numRx*(k-initial_offset)+i,j,complexZero); //set samples at the end to 0+i0
                }
                //Print out the remodulated data estimate to check number of symbol errors (for debugging in Matlab)               
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
    
    //Now make the cross-correlation matrix (rxSigDelays * rxSigDelaysHerm)
    gsl_matrix_complex *autoCorrMatrix = NULL;
    autoCorrMatrix = gsl_matrix_complex_alloc(stackedMatSize,stackedMatSize); //allocate memory
    gsl_complex sampleToConjugate = gsl_complex_rect(1.,0.); //To be used to hold the sample to be conjugate in each loop
    gsl_complex dot_prod = gsl_complex_rect(0.,0.); //declare type for the dot product accumulator
    for(int i = 0; i < stackedMatSize; i++){
        for(int k = 0; k < stackedMatSize; k++){
	    //dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
	    dot_prod = gsl_complex_rect(0.,0.); 
            for(int j = 0; j < num_trn_samp; j++){
        		temp_complex = 	gsl_matrix_complex_get(rxSigDelays,i,j);
        		sampleToConjugate = gsl_matrix_complex_get(rxSigDelays,k,j); //This is the sample to conjugate
        		sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
        		temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate); //Multiply the two samples
			dot_prod = gsl_complex_add(dot_prod,temp_complex); //accumulate (dot product)
            }
            //Place this dot product into the cross correlation matrix
            gsl_matrix_complex_set(autoCorrMatrix,i,k,dot_prod);
    	}
    }
    //end making the auto-correlation matrix
    
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
    gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
    // Compute the  inverse of the LU decomposition
    gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
    printf("just before inversion\n");
    gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
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
    arrayResponse = gsl_matrix_complex_alloc(stackedMatSize,1); //allocate memory 
    //complex matrix multiplication
    gsl_complex unity = gsl_complex_rect(1.,0.); //the complex number 1
    printf("just before making array response.\n");
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, rxSigDelays, x_Training, complexZero, arrayResponse);   
    
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
    beamFormer = gsl_matrix_complex_alloc(stackedMatSize,1); //allocate memory 
    //complex matrix multiplication
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complexZero, beamFormer);    
    printf("just after making beamormer.\n");
    
//     //debugging: check to see if we built the beamformer vector correctly (done) 
//     {
//         FILE * f = fopen ("beamFormer.txt", "w");
//         gsl_matrix_complex_fprintf(f,beamFormer,"%f");
//         fclose(f);
//     }    
    
    
    
/* Now we need to use the computed beamformer vector on the newly received data. The newly received data needs to be
   arranged into a Space-time data matrix rxDataDelays (20x numData) and then perform the operation beamFormer^H * rxDataDelays
*/

//     // grab the received data 
//  //   int numDataSamples = 2048;
//     gsl_matrix_complex *rxData = NULL;
//  //   FILE *rawDataFile3 = fopen(RXDATA,"r");
//     rxData = gsl_matrix_complex_alloc(numRx,num_data_samp);
// //    fread(&dataNumPulses,sizeof(int),1,rawDataFile3);
// //    if((numRx*numDataSamples) != dataNumPulses) {
// //        printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
// //        printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, numRx*numDataSamples);
// //        exit(4);
// //    }
//     for(int i = 0; i < numRx; i++) {
//         for(int j = 0; j < num_data_samp; j++){
//         //    fread(&temp_real, sizeof(double), 1, rawDataFile3);
//         //    fread(&temp_imag, sizeof(double), 1, rawDataFile3);
//             
//           //  temp_complex = gsl_complex_rect(temp_real,temp_imag);
//             gsl_matrix_complex_set(rxData,j,i,gsl_matrix_complex_get(rxSig,i,comms_data_start_idx + j));
//         }
//     }
//  //   fclose(rawDataFile3);
//     printf("Grabbed received data.\r\n");
    
//     //debugging: check to see if we read in the received data correctly  (done)
//     {
//         FILE * f = fopen ("rxData.txt", "w");
//         gsl_matrix_complex_fprintf(f,rxData,"%f");
//         fclose(f);
//     }       
    
 //   //Make delayed matrix (LOS + 4 delay taps), say rxDataDelayed
 //   // Slide rows of rxSig matrix to the left by 1 sample. Fill in with zeros at the end
 //   gsl_matrix_complex *rxDataDelays = NULL;
 //   rxDataDelays = gsl_matrix_complex_alloc(stackedMatSize,numDataSamples); //allocate memory for the space-time matrix
 //       //fill in the space-time matrix using rxSig
 //   for(int i = 0; i < numRx; i++) {
 //       for(int j = 0; j < numDataSamples; j++){
//          //fread(&temp_real, sizeof(double), 1, rawDataFile);
//            //fread(&temp_imag, sizeof(double), 1, rawDataFile);
//            //temp_complex = gsl_complex_rect(temp_real,temp_imag);
//                //place the samples for antenna i for LOS and all delays
//            for(int k = 0; k < 1 + numDelayTaps; k++){
 //               if (j+k < numDataSamples - 1){
//                    temp_complex = gsl_matrix_complex_get(rxData,i,j+k); //the k-th delay sample
//                    gsl_matrix_complex_set(rxDataDelays,(i+1)*numRx*k,j,temp_complex);
//                }
//                else {
//                    gsl_matrix_complex_set(rxDataDelays,(i+1)*numRx*k,j,complexZero); //set samples at the end to 0+i0
//                }
//
//            }
//        }
//    }
    
    
    // Make the delayed matrix, rxSigDelays (LOS + 4 delay taps, say)
    //int numDelayTaps = 4;
    //int stackedMatSize = (1 + numDelayTaps) * numRx;
    //double zero = 0;
    //gsl_complex complexZero = gsl_complex_rect(zero, zero);
    // Slide rows of rxSig matrix to the left by 1 sample. Fill in with zeros at the end
    
    int Nextra = -initial_offset; //we still need to make sure we grab all the way to the end of the data sequence length
    gsl_matrix_complex *rxDataDelays = NULL;
    rxDataDelays = gsl_matrix_complex_alloc(stackedMatSize,num_data_samp + Nextra); //allocate memory for the space-time matrix
    //initial_offset = -1; //begin at the -1th delay tap
    //fill in the space-time matrix using rxSig
    for(int i = 0; i < numRx; i++) { //antenna loop
        for(int j = 0; j < num_data_samp + Nextra; j++){ //time samples loop
            //fread(&temp_real, sizeof(double), 1, rawDataFile);
            //fread(&temp_imag, sizeof(double), 1, rawDataFile);
            //temp_complex = gsl_complex_rect(temp_real,temp_imag);
            //place the samples for antenna i for LOS and all delays
            for(int k = initial_offset; k < Ntaps + initial_offset; k++){ //delay taps loop beginning at -1             
 //               printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                if (j+k < num_data_samp - 1 + initial_offset + Nextra){                 
                    temp_complex = gsl_matrix_complex_get(rxSig,i,comms_data_start_idx + j + k); //the k-th delay sample
                    gsl_matrix_complex_set(rxDataDelays,numRx*(k-initial_offset)+i,j,temp_complex);
                }
                else {
                    gsl_matrix_complex_set(rxDataDelays,numRx*(k-initial_offset)+i,j,complexZero); //set samples at the end to 0+i0
                }
                
            }
        }
    }
    
    printf("Built delayed data matrix.\r\n");
 
    // Multiply w^H * rxDataDelayed
    gsl_matrix_complex *dataEstimate = NULL;
    dataEstimate = gsl_matrix_complex_alloc(1,num_data_samp + Nextra); //allocate memory 
    //complex matrix multiplication
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, rxDataDelays, complexZero, dataEstimate);    

    //MAKE SURE TO DISREGARD THE LAST <Nextra> ENTRY'S OF <dataEstimate> IN ANY FURTHER PROCESSING (i.e. when building the
    //projection matrix) 
    
    printf("Formed beamformed data estimate.\r\n");
    // This gives us our estimate of the transmitted signal (1xnumDataSamples?)
    // Write out estimated signal to file.
    {
        FILE * f = fopen ("data_estimate.txt", "w");
        gsl_matrix_complex_fprintf(f,dataEstimate,"%f");
        fclose(f);
    }

// Now need to map the estimate of what was transmitted back to QAM symbols by
// finding the QAM symbol that is nearest to the estimated symbol
//QAM_constellation 1x4
//dataEstimate 1xnum_data_samp    (make sure to disregard the last Nextra (=1) entries
    gsl_matrix_complex *remodulated_symbols = NULL;
    remodulated_symbols = gsl_matrix_complex_alloc(1,num_data_samp);
    
    //make a vector to temporarily hold the distance metrics
    double temp_dist[N_constellation_points];
    int min_idx;
    gsl_complex temp_complex2 = gsl_complex_rect(1.,0.);
    
    
    for(int i=0;i<num_data_samp;i++)
    {
        temp_complex = gsl_matrix_complex_get(dataEstimate,0,i);
        for(int j=0;j<N_constellation_points;j++)
        {
            temp_complex2 = gsl_matrix_complex_get(QAM_constellation,0,j);
            temp_complex2 = gsl_complex_sub(temp_complex, temp_complex2);
            temp_dist[j] = gsl_complex_abs2(temp_complex2);
        }
        //find minimum entry in temp_dist
        min_idx = 0;
        for(int k=1;k<N_constellation_points;k++)
        {
            if(temp_dist[k] < temp_dist[min_idx])
            {
                min_idx = k; 
            }
        }
        //quanitze the dataEstimate to the nearest QAM symbol
        gsl_matrix_complex_set(dataEstimate,0,i,gsl_matrix_complex_get(QAM_constellation,0,min_idx));
    }
        printf("Remodulated the data estimate.\r\n");
    
        //Print out the remodulated data estimate to check number of symbol errors (for debugging in Matlab)
        {
            FILE * f = fopen ("data_remodulated.txt", "w");
            gsl_matrix_complex_fprintf(f,dataEstimate,"%f");
            fclose(f);
        }
        
// =================== end MMSE comms signal estimation ================================
        
// =================== begin orthogonal projection =====================================
//Now we need to project the received signal onto the subspace orthogonal to the comms signal.
//Beginning at the sync index we project the sync signal, trn signal, and comms data signal.
//IGNORE the sections between these signal where we zero padded the comms signal.
        
        
//=============== begin sync symbols section =========================================
//We already have the real and imaginary components of the sync symbols from the synchronization calculation
        //double sync_symbols_real[numTx,num_sync_samp];
        //double sync_symbols_imag[numTx,num_sync_samp];
        
        //need to type cast each array element to a float (for use in Saquib's temporal mitigation code)
//         sync_symbols_real = (float *)(sync_symbols_real);
//         sync_symbols_imag = (float *)(sync_symbols_imag);
//         
        float *sync_symbols_real_float,*sync_symbols_imag_float;
        
        sync_symbols_real_float = (float *)malloc(numTx*num_sync_samp*sizeof(float));
        sync_symbols_imag_float = (float *)malloc(numTx*num_sync_samp*sizeof(float));
        
        //for(int i=0;i<6;i++) //loop over elements and cast to (float)
        for(int i=0;i<num_sync_samp;i++) //loop over elements and cast to (float)
        {
            sync_symbols_real_float[i] = (float) sync_symbols_real[i];
//             printf("%d -th checking data conversion to float : %f\n",i,sync_symbols_real_float[i]);
//             printf("%d -th from double %f\n\n",i,sync_symbols_real[i]);
            sync_symbols_imag_float[i] = (float) sync_symbols_imag[i];
        }
        

        float float_zero = 0;
        
        float *sync_symbols_real_delay,*sync_symbols_imag_delay;
        sync_symbols_real_delay = (float *)malloc(numTx*Ntaps_projection*num_sync_samp*sizeof(float));
        sync_symbols_imag_delay = (float *)malloc(numTx*Ntaps_projection*num_sync_samp*sizeof(float));
        
        //for loop to build delayed matrix out of *sync_symbols
        ///i = rows in original vector, k = taps, j = time samples
        
        //make the -1th delay signal first (to account for subsample misalignment)
        for(int j=1;j<num_sync_samp;j++)  //the second time sample to the end, plus add a zero after that
        {
            //QUESTION: Does this dynamic memory access automatically account for the size of the data type?
            sync_symbols_real_delay[(j-1)] = sync_symbols_real_float[j];
    //         printf("%d -th sync symbols real delay : %f\n",j-1,sync_symbols_real_delay[(j-1)]);
    //         printf("%d -th sync_symbols_real_float : %f\n",j,sync_symbols_real_float[j]);

            sync_symbols_imag_delay[(j-1)] = sync_symbols_imag_float[j];
            // *(sync_symbols_real_delay + (j-1)) = *(sync_symbols_real + j);
        }
        sync_symbols_real_delay[num_sync_samp - 1] = float_zero;
//Checked this: Good!
//         for(int i=0;i<num_sync_samp;i++)
//         {
//             printf("%d -th sync symbols real delay : %f\n\n",i,sync_symbols_real_delay[i]);            
//         }
        sync_symbols_imag_delay[num_sync_samp - 1] = float_zero;
        //*(sync_symbols_real_delay + num_sync_samp - 1) = float_zero;
        //make the LOS and delayed versions
        
      //  printf("Done with -1th delay tap \n\n");
        
        for(int i=0;i<numTx;i++) //antenna rows
        {
            for(int k = 1;k<Ntaps_projection;k++) //taps
            {
                for(int j=0;j<num_sync_samp;j++) //time samples
                {
//ALL GOOD HERE!!!!               //     printf("k+j = %d + %d = %d\n",k,j,k+j);
               //     printf("sync_symbols_real_delay : %d\n",i*k*num_sync_samp + (k-1) + j);
                //    printf("sync_symbols_real_float : %d\n",j);
                    if((k-1)+j < num_sync_samp)
                    {
                //        printf("In the if\n");
                        //                   printf("i= %d, k =%d, j= %d\n",i,k,j);
                        sync_symbols_real_delay[(i+1)*k*num_sync_samp + (k-1) + j] = sync_symbols_real_float[j]; //PER JACOB
                //               printf("%d -th sync symbols real delay : %f\n",(i+1)*k*num_sync_samp + (k-1) + j,sync_symbols_real_delay[(i+1)*k*num_sync_samp + (k-1) + j]);
                 //              printf("%d -th sync_symbols_real_float : %f\n\n",j,sync_symbols_real_float[j]);
                        sync_symbols_imag_delay[(i+1)*k*num_sync_samp + (k-1) + j] = sync_symbols_imag_float[j];
                        //*((sync_symbols_real_delay + i*k*num_sync_samp) + (k-1) + j) = *(sync_symbols_real + j);
                    }
                    else //need to zero pad at the beginning of the vector
                    {
              //          printf("In the else\n");
                        sync_symbols_real_delay[(i+1)*k*num_sync_samp + ((k-1)-(num_sync_samp - j))] = float_zero;
                        sync_symbols_imag_delay[(i+1)*k*num_sync_samp + ((k-1)-(num_sync_samp - j))] = float_zero;
               //         printf("%d -th sync symbols real delay : %f\n\n",(i+1)*k*num_sync_samp + ((k-1)-(num_sync_samp - j)),sync_symbols_real_delay[(i+1)*k*num_sync_samp + ((k-1)-(num_sync_samp - j))]);
            //            printf("%d -th sync_symbols_real_float : %f\n\n",j,sync_symbols_real_float[j]);
                        // *((sync_symbols_real_delay + i*k*num_sync_samp) + (k-(num_sync_samp - j)) = float_zero;
                    }
                }
                //Checking now...
//                 if(k==1) //This is Good!
//                 {    
//                     for(int iii=0;iii<num_sync_samp;iii++)
//                     {
//                         printf("%d -th sync symbols real delay : %f\n\n",iii,sync_symbols_real_delay[iii]);
//                     }
//                 }
//                 for(int mm=0; mm<num_sync_samp; mm++) //ALL ZEROS!!!! BAD!!!!
//                 {
//                     printf("%d -th sync symbols real delay : %f\n\n",k*num_sync_samp+mm,sync_symbols_real_delay[k*num_sync_samp+mm]);
//                 }
            }
            //   printf("Done with %d -th delay tap \n\n",k);
        }
        
        //MATLAB code
//         S(1,:) = [remod_s_tx_comms(1,2:end) 0];
//         for ii = 1:Ndel - 1
//                 S(ii+1,:) = [zeros(1,ii-1) remod_s_tx_comms(1,1:end-(ii-1))];
//         end
        
        
//Get the relevant section of the received data (according to our calculated sync index)
//   comms_sync_start_idx = ipeak;
//   comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros;
//   comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;
        
        //ORIGINAL RX DATA MATRIX : rxSig = gsl_matrix_complex_alloc(numRx,num_rx_samp);
        float *rx_sync_real,*rx_sync_imag;
        
        rx_sync_real = (float *)malloc(numRx*num_sync_samp*sizeof(float));
        rx_sync_imag = (float *)malloc(numRx*num_sync_samp*sizeof(float));
        
        for(int i=0;i<numRx;i++)
        {
            for(int j=0;j<num_sync_samp;j++)
            {
                rx_sync_real[i*num_sync_samp + j] = GSL_REAL(gsl_matrix_complex_get(rxSig,i,comms_sync_start_idx + j));
                rx_sync_imag[i*num_sync_samp + j] = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,comms_sync_start_idx + j));
            }
        }
        
//=============== end sync symbols section =========================================
        
//=============== begin training symbols section =========================================
        float *trn_symbols_real_delay,*trn_symbols_imag_delay;
        
        trn_symbols_real_delay = (float *)malloc(numTx*Ntaps_projection*num_trn_samp*sizeof(float));
        trn_symbols_imag_delay = (float *)malloc(numTx*Ntaps_projection*num_trn_samp*sizeof(float));
        
        //for loop to build delayed matrix out of *x_Training
        ///i = rows in original vector, k = taps, j = time samples
        
        //make the -1th delay signal first (to account for subsample misalignment)
        for(int j=1;j<num_trn_samp;j++)  //the second time sample to the end, plus add a zero after that
        {
            //QUESTION: Does this dynamic memory access automatically account for the size of the data type?
            trn_symbols_real_delay[(j-1)] = GSL_REAL(gsl_matrix_complex_get(x_Training,0,j));
            trn_symbols_imag_delay[(j-1)] = GSL_IMAG(gsl_matrix_complex_get(x_Training,0,j));
        }
        trn_symbols_real_delay[num_trn_samp - 1] = float_zero;
        trn_symbols_imag_delay[num_trn_samp - 1] = float_zero;
        
        //make the LOS and delayed versions
        for(int i=0;i<numTx;i++) //rows
        {
            for(int k = 1;k<Ntaps_projection;k++) //taps
            {
                for(int j=0;j<num_trn_samp;j++) //time samples
                {
                    if((k-1)+j < num_trn_samp)
                    {
                        trn_symbols_real_delay[(i+1)*k*num_trn_samp + (k-1) + j] = GSL_REAL(gsl_matrix_complex_get(x_Training,0,j));
                        trn_symbols_imag_delay[(i+1)*k*num_trn_samp + (k-1) + j] = GSL_IMAG(gsl_matrix_complex_get(x_Training,0,j));
                    }
                    else //need to zero pad at the beginning of the vector
                    {
                        trn_symbols_real_delay[(i+1)*k*num_trn_samp + ((k-1)-(num_trn_samp - j))] = float_zero;
                        trn_symbols_imag_delay[(i+1)*k*num_trn_samp + ((k-1)-(num_trn_samp - j))] = float_zero;
                    }
                }
            }
        }
        
        
//Get the relevant section of the received data (according to our calculated sync index)
//   comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros;
        
        //ORIGINAL RX DATA MATRIX : rxSig = gsl_matrix_complex_alloc(numRx,num_rx_samp);
        float *rx_trn_real,*rx_trn_imag;
        
        rx_trn_real = (float *)malloc(numRx*num_trn_samp*sizeof(float));
        rx_trn_imag = (float *)malloc(numRx*num_trn_samp*sizeof(float));
        
        for(int i=0;i<numRx;i++)
        {
            for(int j=0;j<num_trn_samp;j++)
            {
                rx_trn_real[i*num_trn_samp + j] = GSL_REAL(gsl_matrix_complex_get(rxSig,i,comms_trn_start_idx + j));
                rx_trn_imag[i*num_trn_samp + j] = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,comms_trn_start_idx + j));
            }
        }   
//=============== end training symbols section =========================================
        
//=============== begin data symbols section =========================================
        float *data_remod_real_delay,*data_remod_imag_delay;
        
        data_remod_real_delay = (float *)malloc(numTx*Ntaps_projection*num_data_samp*sizeof(float));
        data_remod_imag_delay = (float *)malloc(numTx*Ntaps_projection*num_data_samp*sizeof(float));
        
        //for loop to build delayed matrix out of *dataEstimate  (the remodulated data estimate)
        ///i = rows in original vector, k = taps, j = time samples
        
        //make the -1th delay signal first (to account for subsample misalignment)
        for(int j=1;j<num_data_samp;j++)  //the second time sample to the end, plus add a zero after that
        {
            //QUESTION: Does this dynamic memory access automatically account for the size of the data type?
            data_remod_real_delay[(j-1)] = GSL_REAL(gsl_matrix_complex_get(dataEstimate,0,j));
            data_remod_imag_delay[(j-1)] = GSL_IMAG(gsl_matrix_complex_get(dataEstimate,0,j));
        }
        data_remod_real_delay[num_data_samp - 1] = float_zero;
        data_remod_imag_delay[num_data_samp - 1] = float_zero;
        
        //make the LOS and delayed versions
        for(int i=0;i<numTx;i++) //rows
        {
            for(int k = 1;k<Ntaps_projection;k++) //taps
            {
                for(int j=0;j<num_data_samp;j++) //time samples
                {
                    if((k-1)+j < num_data_samp)
                    {
                        data_remod_real_delay[(i+1)*k*num_data_samp + (k-1) + j] = GSL_REAL(gsl_matrix_complex_get(dataEstimate,0,j));
                        data_remod_imag_delay[(i+1)*k*num_data_samp + (k-1) + j] = GSL_IMAG(gsl_matrix_complex_get(dataEstimate,0,j));
                    }
                    else //need to zero pad at the beginning of the vector
                    {
                        data_remod_real_delay[(i+1)*k*num_data_samp + ((k-1)-(num_data_samp - j))] = float_zero;
                        data_remod_imag_delay[(i+1)*k*num_data_samp + ((k-1)-(num_data_samp - j))] = float_zero;
                    }
                }
            }
        }
        
        
//Get the relevant section of the received data (according to our calculated sync index)
//   comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;
        
        //ORIGINAL RX DATA MATRIX : rxSig = gsl_matrix_complex_alloc(numRx,num_rx_samp);
        float *rx_data_real,*rx_data_imag;
        
        rx_data_real = (float *)malloc(numRx*num_data_samp*sizeof(float));
        rx_data_imag = (float *)malloc(numRx*num_data_samp*sizeof(float));
        
        for(int i=0;i<numRx;i++)
        {
            for(int j=0;j<num_data_samp;j++)
            {
                rx_data_real[i*num_data_samp + j] = GSL_REAL(gsl_matrix_complex_get(rxSig,i,comms_data_start_idx + j));
                rx_data_imag[i*num_data_samp + j] = GSL_IMAG(gsl_matrix_complex_get(rxSig,i,comms_data_start_idx + j));
            }
        }    
//=============== end data symbols section =========================================
        
//Project the relevant section of the received data onto the subspace orthogonal to the comms signal.
//Need to cycle over loops of 64 time samples.
 
//outputs: to be used on every for loop iteration       
    float *zres , *zresimag;
    zres = (float *)malloc(N*M*sizeof(float));
    zresimag = (float *)malloc(N*M*sizeof(float));
  
//         int modulo_N = 64; //We need to call Saquib's temporal mitigation code in blocks of 64 time samples
//     int n_sync_blocks = 4; //number of comms sync blocks of 64 time samples
//     int n_trn_blocks = 4; //number of comms training blocks of 64 time samples
//     int n_data_blocks = 280;
    
    //intialize matrices to hold the data for calling Saquib's code
    float *temp_S_real, *temp_S_imag, *temp_Z_real, *temp_Z_imag;
    //allocate space
    temp_S_real = (float *)malloc(Ntaps_projection*modulo_N*sizeof(float));
    temp_S_imag = (float *)malloc(Ntaps_projection*modulo_N*sizeof(float));
    temp_Z_real = (float *)malloc(numRx*modulo_N*sizeof(float));
    temp_Z_imag = (float *)malloc(numRx*modulo_N*sizeof(float));
    
    //Sync symbols section
    int k = 0;
    while(k< (num_sync_samp)) //total number of sync time samples
    {
        for(int i=0;i<Ntaps_projection;i++) //rows
        {
            for(int j=k;j<k+modulo_N;j++) //time samples, jumping by blocks
            {
//ALL GOOD HERE!!!         //       printf("j : %d\n",j);
                temp_S_real[i*modulo_N + (j-k)] = sync_symbols_real_delay[i*num_sync_samp + j];
                temp_S_imag[i*modulo_N + (j-k)] = sync_symbols_imag_delay[i*num_sync_samp + j];
                temp_Z_real[i*modulo_N + (j-k)] = rx_sync_real[i*num_sync_samp + j];
                temp_Z_imag[i*modulo_N + (j-k)] = rx_sync_imag[i*num_sync_samp + j];
          //      printf("%d entry of temp_S_real: %f\n",i*modulo_N + (j-k),temp_S_real[i*modulo_N + (j-k)]);
          //      printf("%d -th sync_symbols_real_delay : %f\n\n",i*num_sync_samp + j,sync_symbols_real_delay[i*num_sync_samp + j]);
                
            }
        }
        
//ALL GOOD HERE!!! //         if(k==0)
//         {    
//             for(int ll=0;ll<num_sync_samp;ll++)
//             {
//                 printf("%d -th element of sync real is : %f\n\n",ll,temp_S_real[ll]);
//             }
//         }
            //debugging: check to see if we can do projection in MATLAB
//        if(k==64)
//        {
//         FILE * freal = fopen ("sync_projection_real.txt", "w");
//        for(int ll=0;ll<num_sync_samp;ll++)
//        {
//             fprintf(freal,"%f\r\n",temp_S_real[ll]);   
//        }
//        fclose(freal);
//                FILE * fimag = fopen ("sync_projection_imag.txt", "w");
//        for(int ll=0;ll<num_sync_samp;ll++)
//        {
//             fprintf(fimag,"%f\r\n",temp_S_imag[ll]);   
//        }
//        fclose(fimag);
//        }
//     //call Saquib's temporal mitigation code
//     printf("\nCalling temporal_mitigation for sync symbols block: %d\n",k/modulo_N + 1);
//     temporal_mitigation(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);
    printf("\nCalling QR factorization projection method for sync symbols block: %d\n",k/modulo_N + 1);
    QR_factorization_projection(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);
        //replace our original received data with this newly projected data
        for(int ii=0;ii<numRx;ii++) //loop over each receive antenna (rows of rxSig)
        {
            for(int jj=0;jj<modulo_N;jj++) //loop over the number of samples in this newly projected data vector
            {

//   comms_sync_start_idx = ipeak;    
                temp_real = zres[jj];
                temp_imag = zresimag[jj];
                temp_complex = gsl_complex_rect(temp_real,temp_imag);
                //place at the time sample  [ comms_sync_start_idx + this block (k) + this time sample (jj) ]
                gsl_matrix_complex_set(rxSig,ii,comms_sync_start_idx + k + jj,temp_complex);
            }
        } //END replacing our original received data with the newly projected data
        
        //update k by the length of a block (64 times samples)
        k+=modulo_N;
    } //END while

        

        //Training symbols section
    k = 0;
    while(k< (num_trn_samp)) //total number of training time samples
    {
        for(int i=0;i<Ntaps_projection;i++) //rows
        {
            for(int j=k;j<k+modulo_N;j++) //time samples, jumping by blocks
            {
                temp_S_real[i*modulo_N + (j-k)] = trn_symbols_real_delay[i*num_trn_samp + j];
                temp_S_imag[i*modulo_N + (j-k)] = trn_symbols_imag_delay[i*num_trn_samp + j];
                temp_Z_real[i*modulo_N + (j-k)] = rx_trn_real[i*num_trn_samp + j];
                temp_Z_imag[i*modulo_N + (j-k)] = rx_trn_imag[i*num_trn_samp + j];
                
            }
        }
//                 printf("\nCalling temporal_mitigation for training symbols block: %d\n",k/modulo_N + 1);
//         //call Saquib's temporal mitigation code
//         temporal_mitigation(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);
    printf("\nCalling QR factorization projection method for training symbols block: %d\n",k/modulo_N + 1);
    QR_factorization_projection(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);        
//replace our original received data with this newly projected data
        for(int ii=0;ii<numRx;ii++) //loop over each receive antenna (rows of rxSig)
        {
            for(int jj=0;jj<modulo_N;jj++) //loop over the number of samples in this newly projected data vector
            {
//   comms_trn_start_idx = ipeak + num_sync_samp + n_sync_trn_zeros;      
                temp_real = zres[jj];
                temp_imag = zresimag[jj];
                temp_complex = gsl_complex_rect(temp_real,temp_imag);
                //place at the time sample  [ comms_trn_start_idx + this block (k) + this time sample (jj) ]
                gsl_matrix_complex_set(rxSig,ii,comms_trn_start_idx + k + jj,temp_complex);
            }
        } //END replacing our original received data with the newly projected data
        
        //update k by the length of a block (64 times samples)
        k+=modulo_N;
    } //END while
            

        //Data symbols section
    k = 0;
    while(k< (num_data_samp)) //total number of data time samples
    {
        for(int i=0;i<Ntaps_projection;i++) //rows
        {
            for(int j=k;j<k+modulo_N;j++) //time samples, jumping by blocks
            {
                temp_S_real[i*modulo_N + (j-k)] = data_remod_real_delay[i*num_data_samp + j];
                temp_S_imag[i*modulo_N + (j-k)] = data_remod_imag_delay[i*num_data_samp + j];
                temp_Z_real[i*modulo_N + (j-k)] = rx_data_real[i*num_data_samp + j];
                temp_Z_imag[i*modulo_N + (j-k)] = rx_data_imag[i*num_data_samp + j];
                
            }
        }
        
//         if(k==0)
//         {
//             FILE * freal = fopen ("data_projection_real.txt", "w");
//             for(int ll=0;ll<num_sync_samp;ll++)
//             {
//                 fprintf(freal,"%f\r\n",temp_S_real[ll]);
//             }
//             fclose(freal);
//             FILE * fimag = fopen ("data_projection_imag.txt", "w");
//             for(int ll=0;ll<num_sync_samp;ll++)
//             {
//                 fprintf(fimag,"%f\r\n",temp_S_imag[ll]);
//             }
//             fclose(fimag);
//         }
        
//               printf("\nCalling temporal_mitigation for data symbols block: %d\n",k/modulo_N + 1);
//         //call Saquib's temporal mitigation code
//         temporal_mitigation(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);
            printf("\nCalling QR factorization projection method for data symbols block: %d\n",k/modulo_N + 1);
    QR_factorization_projection(temp_Z_real, temp_Z_imag, temp_S_real, temp_S_imag, zres, zresimag);
        //replace our original received data with this newly projected data
        for(int ii=0;ii<numRx;ii++) //loop over each receive antenna (rows of rxSig)
        {
            for(int jj=0;jj<modulo_N;jj++) //loop over the number of samples in this newly projected data vector
            {
//   comms_data_start_idx = comms_trn_start_idx + num_trn_samp + num_trn_data_zeros;       
                temp_real = zres[jj];
                temp_imag = zresimag[jj];
                temp_complex = gsl_complex_rect(temp_real,temp_imag);
                //place at the time sample  [ comms_data_start_idx + this block (k) + this time sample (jj) ]
                gsl_matrix_complex_set(rxSig,ii,comms_data_start_idx + k + jj,temp_complex);
            }
        } //END replacing our original received data with the newly projected data
        
        //update k by the length of a block (64 times samples)
        k+=modulo_N;
    } //END while

       
//Send rxSig on to further radar processing 
    
    
    
// =================== end orthogonal projection =====================================
    
    return 0;
// END RF_convergence.c
}

// ******************* Support functions ***********************


//=========================== sync_MMSE ======================================================

//Inputs:   numRx, numTx, Ntaps, num_sync_samp, num_rx_samp, sync_symbols (real,imaginary),
//          rx_data (real,imaginary)
//output: the starting time sample index for the comms signal (ipeak = 400)
int sync_MMSE(int numRx, int numTx, int Ntaps, int num_sync_samp, int num_rx_samp, double *sync_symbols_real, double *sync_symbols_imag, double *rx_data_sync_real, double *rx_data_sync_imag)
{
    
    double threshold = .7; //the sync statistic will be between [0,1], .7 is an educated guess
//     int numRx = 4; //number of receive antennas
//     int numTx = 1; //number of transmit antennas
//     int Ntaps = 5; //number of delay taps
    int delays[Ntaps]; //vector of taps
    
    for(int k=0; k<Ntaps; k++)
    {
        delays[k] = k;
    }
    
    //Build a weighting vector for the MMSE beamformer. Build it as a decreasing step function where
    //each step corresponds to a new delay tap. This is necessary to give us a sharp, unique peak for our
    //sync statistic.
    double Ntaps_double =(double)Ntaps; //change Ntaps from int to double
    double sync_weight_taper[numRx*Ntaps];
    for(int k=0; k<Ntaps; k++)
    {
        for(int j=0; j<numRx; j++)
        {
            sync_weight_taper[k*numRx + j] = (Ntaps_double - k) / Ntaps_double; //e.g. 1, 4/5, 3/5, 2/5, 1/5
        }
    }
    
//     //Read in the received data
//     int numRxSamples = 2000;
    gsl_matrix_complex *rxSig = NULL;
//     FILE *rawDataFile = fopen(RX,"r");
    rxSig = gsl_matrix_complex_alloc(numRx,num_rx_samp);
    double temp_real,temp_imag;
    gsl_complex temp_complex = gsl_complex_rect(1.,0.);
//     int dataNumPulses;
//     fread(&dataNumPulses,sizeof(int),1,rawDataFile);
//     if((numRx*numRxSamples) != dataNumPulses) {
//         printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
//         printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, numRx*numRxSamples);
//         exit(4);
//     }
    for(int i = 0; i < numRx; i++)
    {
        for(int j = 0; j < num_rx_samp; j++)
        {
//             fread(&temp_real, sizeof(double), 1, rawDataFile);
//             fread(&temp_imag, sizeof(double), 1, rawDataFile);
            temp_real = *( (rx_data_sync_real+ i*num_rx_samp) + j);
            temp_imag = *( (rx_data_sync_imag+ i*num_rx_samp) + j);
            temp_complex = gsl_complex_rect(temp_real,temp_imag);
            gsl_matrix_complex_set(rxSig,i,j,temp_complex);
        }
    }
//     fclose(rawDataFile);
    
    //Read in the sync training sequence
    //   int numTxSamples = 256;
    gsl_matrix_complex *syncSig = NULL;
    //   FILE *rawDataFile2 = fopen(SYNC_TRN,"r");
    syncSig = gsl_matrix_complex_alloc(numTx,num_sync_samp);
//    fread(&dataNumPulses,sizeof(int),1,rawDataFile);
//    if((numTx*numTxSamples) != dataNumPulses) {
//        printf("Number of pulses specified by config file and raw data file differ! Are you sure this data matches your config?\r\n");
//        printf("dataNumPulses: %d, numPulses: %d\r\n",dataNumPulses, numTx*numTxSamples);
//        exit(4);
//    }
    for(int i = 0; i < numTx; i++) {
        for(int j = 0; j < num_sync_samp; j++){
            //   fread(&temp_real, sizeof(double), 1, rawDataFile);
            //   fread(&temp_imag, sizeof(double), 1, rawDataFile);
            temp_real = *( (sync_symbols_real+ i*num_sync_samp) + j);
            temp_imag = *( (sync_symbols_imag+ i*num_sync_samp) + j);
            temp_complex = gsl_complex_rect(temp_real,temp_imag);
            gsl_matrix_complex_set(syncSig,i,j,temp_complex);
        }
    }
    //   fclose(rawDataFile);
    
    
    //initialize variables
    int sync_flag = 0; //set to 1 when we have found a possible sync index
    int NsyncStatistics = num_rx_samp - num_sync_samp - 1;
    double sync_statistics[NsyncStatistics]; //to hold our generated sync statistics
    for(int k=0;k<NsyncStatistics;k++)
    {
        sync_statistics[k] = 0;
    }
//    double divisionResult;
    int index = 0; //used for counting our sync statistics
    int ipeak = 0; //this will be our detected sync sample index
    double ipeakVal = 0.0;  //this will be our peak sync statistic value
    
    double temp_norm_sum = 0.0;
    //gsl_complex temp_normalizer = gsl_complex_rect(0.,0.);
    gsl_complex unity = gsl_complex_rect(1.,0.); //the complex number 1
    gsl_complex complex_zero = gsl_complex_rect(0.,0.); //the complex number 0
    
    //the normilizer should be the norm of the transmitted sync training sequence
    // double normalizer = 256.0;  //
    
    
    //calculate the norm of the sync training sequence to be used for normalizing our sync statistics
    //temp_norm_sum = 0;
    // printf("temp_norm_sum before loop: %f\r\n\n",temp_norm_sum);
    for(int k=0;k<num_sync_samp;k++)
        //for(int k=0;k<4;k++)
    {
        //================================================================
        //Debugging...the abs^2 of the syncEstimate grows on each iteration.
        //================================================================
        
        //printf("temp_real at start of loop: %f\r\n",temp_real);
        temp_complex = gsl_matrix_complex_get(syncSig,0,k);
        //sampleToConjugate = gsl_complex_conjugate(temp_complex);
        //temp_complex = gsl_complex_mul(temp_complex,sampleToConjugate);
        temp_real = gsl_complex_abs2(temp_complex);
        //printf("prior temp_norm_sum inside of loop: %f\r\n\n",temp_norm_sum);
        // printf("temp_real before sum inside of loop: %f\r\n",temp_real);
        temp_norm_sum = temp_norm_sum + temp_real;
        //temp_complex = gsl_complex_rect(0.,0.);
        //temp_real = 0;
        //printf("post temp_norm_sum inside of loop: %f\r\n\n",temp_norm_sum);
    }
    double normalizer;
    normalizer = temp_norm_sum; //set the normalizer variable
    temp_norm_sum = 0; //zero out for later use
    
    //Begin algorithm
    //will hold the relevant section of received data for calculating each test statistic
    gsl_matrix_complex *temp_Rx_data = NULL;
    temp_Rx_data = gsl_matrix_complex_alloc(numRx,num_sync_samp);
    //will hold the space-time delay received data
    gsl_matrix_complex *temp_Rx_space_time_data = NULL;
    temp_Rx_space_time_data = gsl_matrix_complex_alloc(numRx*Ntaps,num_sync_samp);
    
    
    //====================
    //initialize all variables to be used in the sync algorithm loops
    int stackedMatSize = numRx*Ntaps;
    gsl_matrix_complex *autoCorrMatrix = NULL;
    autoCorrMatrix = gsl_matrix_complex_alloc(stackedMatSize,stackedMatSize); //allocate memory
    gsl_complex sampleToConjugate = gsl_complex_rect(1.,0.); //To be used to hold the sample to be conjugate in each loop
    gsl_complex dot_prod = gsl_complex_rect(0.,0.); //declare type for the dot product accumulator
    
    gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
    int s;
    
    gsl_matrix_complex *arrayResponse = NULL;
    arrayResponse = gsl_matrix_complex_alloc(stackedMatSize,1); //allocate memory
    
    gsl_matrix_complex *beamFormer = NULL;
    beamFormer = gsl_matrix_complex_alloc(stackedMatSize,1); //allocate memory
    
    gsl_complex temp_complex_taper = gsl_complex_rect(0.,0.); //will be used to hold the taper number
    double real_taper = 0; //allocating space
    gsl_complex temp_product = gsl_complex_rect(0.,0.); //will temporarily hold the tapered beamformer weight
    
    gsl_matrix_complex *syncEstimate = NULL;
    syncEstimate = gsl_matrix_complex_alloc(numTx,num_sync_samp);
    
    gsl_matrix_complex *temp_complex_matrix = NULL;
    temp_complex_matrix = gsl_matrix_complex_alloc(1,1); //scalar as complex matrix data type
    
//             // For debugging.
//         {
//             FILE * f = fopen ("Zin.txt", "w");
//             gsl_matrix_complex_fprintf(f,rxSig,"%f");
//             fclose(f);
//         }
    
    //begin iterating over the sync algorithm
    for(int ll=0; ll<NsyncStatistics; ll++)
    {
        index = ll;
        //get the relevant received data for calculating the test statistic for time sample 'index'
        for(int i = 0; i < numRx; i++) //antenna loop
        {
            for(int j = 0; j < num_sync_samp; j++) //time samples loop
            {
                temp_complex = gsl_matrix_complex_get(rxSig,i,index + j); //the index + j-th received sample
                gsl_matrix_complex_set(temp_Rx_data,i,j,temp_complex);
            }
        }
        
//         // For debugging
//         {
//             FILE * f = fopen ("temp_Rx_data.txt", "w");
//             gsl_matrix_complex_fprintf(f,temp_Rx_data,"%f");
//             fclose(f);
//         }
        
        //build the space-time delay receive matrix
        for(int i = 0; i < numRx; i++)
        { //antenna loop
            for(int j = 0; j < num_sync_samp; j++)
            { //time samples loop
                for(int k = 0; k < Ntaps; k++)
                { //delay taps loop
                    //printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                    if (j+k <= num_sync_samp - 1)
                    {
                        temp_complex = gsl_matrix_complex_get(temp_Rx_data,i,j+k); //the k-th delay sample
                        gsl_matrix_complex_set(temp_Rx_space_time_data,numRx*k+i,j,temp_complex);
                    }
                    else
                    {
                        gsl_matrix_complex_set(temp_Rx_space_time_data,numRx*k+i,j,complex_zero); //set samples at the end to 0+i0
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
        
        //build auto-correlation matrix and then invert it
        for(int i = 0; i < stackedMatSize; i++)
        {
            for(int k = 0; k < stackedMatSize; k++)
            {
                dot_prod = gsl_complex_rect(0.,0.); //reset the dot-product variable to zero
                //dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
                for(int j = 0; j < num_sync_samp; j++)
                {
                    temp_complex = 	gsl_matrix_complex_get(temp_Rx_space_time_data,i,j);
                    sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data,k,j); //This is the sample to conjugate
                    sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
                    temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate); //Multiply the two samples
                    dot_prod = gsl_complex_add(dot_prod,temp_complex); //accumulate (dot product)
                }
                //Place this dot product into the cross correlation matrix
                gsl_matrix_complex_set(autoCorrMatrix,i,k,dot_prod);
            }
        }
        
//                 //For debugging
//         {
//             FILE * f = fopen ("autoCorr.txt", "w");
//             gsl_matrix_complex_fprintf(f,autoCorrMatrix,"%f");
//             fclose(f);
//         }
        
        //end making the auto-correlation matrix
        
        //invert auto-correlation matrix
        
        // Compute the LU decomposition of this matrix
        gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
        // Compute the  inverse of the LU decomposition
        gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
        gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
        
//                     // For debugging
//         {
//             FILE * f = fopen ("invAutoCorr.txt", "w");
//             gsl_matrix_complex_fprintf(f,invAutoCorr,"%f");
//             fclose(f);
//         }
        
        // Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H
        gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero, arrayResponse);
        
//         //For debugging
//         {
//             FILE * f = fopen ("arrayResponse.txt", "w");
//             gsl_matrix_complex_fprintf(f,arrayResponse,"%f");
//             fclose(f);
//         }
        
        // Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
        // This gives us our beamforming vector w
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
        
//                             // For debugging
//         {
//             FILE * f = fopen ("beamFormer.txt", "w");
//             gsl_matrix_complex_fprintf(f,beamFormer,"%f");
//             fclose(f);
//         }
        
// taper the beamformer weights
        for(int k=0; k<numRx*Ntaps; k++)
        {
            //get the beamformer entry
            temp_complex = gsl_matrix_complex_get(beamFormer, k,0);
            //get the taper entry
            real_taper = sync_weight_taper[k];
            temp_complex_taper = gsl_complex_rect(real_taper,0.);
            temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
            gsl_matrix_complex_set(beamFormer, k,0, temp_product);
            
        }
        
//     //For debugging
//     {
//         FILE * f = fopen ("beamFormer_tapered.txt", "w");
//         gsl_matrix_complex_fprintf(f,beamFormer,"%f");
//         fclose(f);
//     }
        
        //Compute the MMSE estimate of what was sent given this data
        gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero, syncEstimate);
        
//     //For debugging
//     {
//         FILE * f = fopen ("syncEstimate.txt", "w");
//         gsl_matrix_complex_fprintf(f,syncEstimate,"%f");
//         fclose(f);
//     }
        
//calculate the norm of the sync estimate
        temp_norm_sum = 0;
        for(int k=0;k<num_sync_samp;k++)
        {
            temp_complex = gsl_matrix_complex_get(syncEstimate,0,k);
            temp_real = gsl_complex_abs2(temp_complex);
            temp_norm_sum = temp_norm_sum + temp_real;
            temp_complex = gsl_complex_rect(0.,0.);
            temp_real = 0;
        }
        //normalize and set the sync statistic
        sync_statistics[index] = temp_norm_sum / normalizer;
        temp_norm_sum = 0; //set to zero for use on next iteration
        
        //check to see if we crossed the threshold
        if(sync_statistics[index] > threshold)
        {
            ipeak = index;
            ipeakVal = sync_statistics[index];
            break;
        }
        
    } //end of algorithm FOR loop
    
    
    //We have crossed the threshold (or never did), now we need to check if the time indices immediately after have a higher sync statistic
    if (ipeak == 0) //we never synced, set ipeak and ipeakVal to -1 to alert us
    {
        ipeak = -1;
        ipeakVal = -1;
    }
    
    //Do the same sync algorithm on the next time indices until we see the first decline in value. Then we have our hypothsised sync index
    while(sync_flag == 0 && ipeak != -1)
    {
        index = index + 1;
        
        for(int i = 0; i < numRx; i++) //antenna loop
        {
            for(int j = 0; j < num_sync_samp; j++) //time samples loop
            {
                temp_complex = gsl_matrix_complex_get(rxSig,i,index + j); //the index + j-th received sample
                gsl_matrix_complex_set(temp_Rx_data,i,j,temp_complex);
            }
        }
        
        //build the space-time delay receive matrix
        for(int i = 0; i < numRx; i++)
        { //antenna loop
            for(int j = 0; j < num_sync_samp; j++)
            { //time samples loop
                for(int k = 0; k < Ntaps; k++)
                { //delay taps loop
                    //printf("I: %d, J: %d, K: %d\r\n",i,j,k);
                    if (j+k <= num_sync_samp - 1)
                    {
                        temp_complex = gsl_matrix_complex_get(temp_Rx_data,i,j+k); //the k-th delay sample
                        gsl_matrix_complex_set(temp_Rx_space_time_data,numRx*k+i,j,temp_complex);
                    }
                    else
                    {
                        gsl_matrix_complex_set(temp_Rx_space_time_data,numRx*k+i,j,complex_zero); //set samples at the end to 0+i0
                    }
                }
            }
        }
        
        for(int i = 0; i < stackedMatSize; i++)
        {
            for(int k = 0; k < stackedMatSize; k++)
            {
                //dot product of the i_th row of rxSigDelays with the k_th column of rxSigDelays^Herm
                for(int j = 0; j < num_sync_samp; j++)
                {
                    temp_complex = 	gsl_matrix_complex_get(temp_Rx_space_time_data,i,j);
                    sampleToConjugate = gsl_matrix_complex_get(temp_Rx_space_time_data,k,j); //This is the sample to conjugate
                    sampleToConjugate = gsl_complex_conjugate(sampleToConjugate);
                    temp_complex = gsl_complex_mul(temp_complex, sampleToConjugate); //Multiply the two samples
                    dot_prod = gsl_complex_add(dot_prod,temp_complex); //accumulate (dot product)
                }
                //Place this dot product into the cross correlation matrix
                gsl_matrix_complex_set(autoCorrMatrix,i,k,dot_prod);
                dot_prod = gsl_complex_rect(0.,0.); //reset the dot-product variable to zero
            }
        }
        //end making the auto-correlation matrix
        
        //invert auto-correlation matrix
        gsl_permutation *p = gsl_permutation_alloc(stackedMatSize);
        // Compute the LU decomposition of this matrix
        gsl_linalg_complex_LU_decomp(autoCorrMatrix, p, &s);
        // Compute the  inverse of the LU decomposition
        gsl_matrix_complex *invAutoCorr = gsl_matrix_complex_alloc(stackedMatSize, stackedMatSize);
        gsl_linalg_complex_LU_invert(autoCorrMatrix, p, invAutoCorr);
        //      gsl_permutation_free(p);
        
        // Matrix multiply arrayResponse = temp_Rx_space_time_data * syncSig^H
        gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, unity, temp_Rx_space_time_data, syncSig, complex_zero, arrayResponse);
        
        
        // Matrix mulitpy  (R*R^Hermitian)^-1 * (arrayResponse)
        // This gives us our beamforming vector w
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, unity, invAutoCorr, arrayResponse, complex_zero, beamFormer);
        
// taper the beamformer weights
        for(int k=0; k<numRx*Ntaps; k++)
        {
            //get the beamformer entry
            temp_complex = gsl_matrix_complex_get(beamFormer, k,0);
            //get the taper entry
            real_taper = sync_weight_taper[k];
            temp_complex_taper = gsl_complex_rect(real_taper,0.);
            temp_product = gsl_complex_mul(temp_complex, temp_complex_taper);
            gsl_matrix_complex_set(beamFormer, k,0, temp_product);
            
        }
        
        //Compute the MMSE estimate of what was sent given this data
        gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, unity, beamFormer, temp_Rx_space_time_data, complex_zero, syncEstimate);
        
        //calculate the norm of the sync estimate
        for(int k=0;k<num_sync_samp;k++)
        {
            temp_complex = gsl_matrix_complex_get(syncEstimate,0,k);
            temp_real = gsl_complex_abs2(temp_complex);
            temp_norm_sum = temp_norm_sum + temp_real;
            temp_complex = gsl_complex_rect(0.,0.);
            temp_real = 0;
        }
        
        //normalize and set the sync statistic
        sync_statistics[index] = temp_norm_sum / normalizer;
        temp_norm_sum = 0; //set to zero for use on next iteration
        
        //check to see if our sync statistic is increasing, if so, then replace our peak index
        if(sync_statistics[index] > sync_statistics[index - 1])
        {
            
            ipeak = index;
            ipeakVal = sync_statistics[index];
        }
        else    //our sync statistic decreased, thus we already have our peak index. We are done!
        {
            sync_flag = 1;
            gsl_permutation_free(p);
        }
        
    }
    
//  printf("The final sync index is: %d \n", ipeak);
//  printf("The final sync peak value is: %f \n", ipeakVal);
//    printf("%d,%f",ipeak,ipeakVal);
    
    return ipeak; //return the index representing the begining of our sync training sequence
} //Done with sync_MMSE support function

//===============================================================================================


//================ Matt's QR-factorization approach ================================================

void QR_factorization_projection(float *temp_Z_real, float *temp_Z_imag, float *temp_S_real, float *temp_S_imag, float *zres_real, float *zres_imag)
{
//(Band-aid) Try to implement the temporal mitigation using the QR factorization approach
//  Inputs:     Z: numRx x numSamp (4x64) received data that we wish to project onto the subspace orthogonal to
//                  S.
//              S: The comms signal that we will use to build the projection matrix
//  
//  Outputs:    Zres: The received data after temporal projection onto the subspace orthogonal to S
//    
    
    
    
}

//================ End QR-factorization approach ================================================





//======== Saquib's temporal mitigation ========

// #include "inverse.h"
// 
// #define PROGPATH DASH_DATA "Dash-RadioCorpus/temporal_mitigation/"
// #define ZIN PROGPATH "z.txt"
// #define ZIMAGIN PROGPATH "zimag.txt"
// #define SIN PROGPATH "s.txt"
// #define SIMAGIN PROGPATH "simag.txt"

// class perf_counter
// {
//     public:
//         uint64_t tot, cnt, calls;
//         perf_counter() : tot(0), cnt(0), calls(0) {};
//         inline void reset() { tot = cnt = calls = 0; }
//         inline void start() { cnt = 0; calls++; };
//         inline void stop() { tot +=  (0 - cnt); };
//         inline uint64_t avg_cpu_cycles() { return ((tot+(calls>>1)) / calls); }
// };


//Maybe we could use this function to place the projected version of the received signal back into the 
//address location of the un-projected received signal Z,Zi????

void temporal_mitigation(float *Z, float *Zi, float *S, float *Si, float *zres, float *zresimag)
{
    ////Initializing the main variables that will be required in the overall computation of the Z response
//    perf_counter hw_ctr,hw_ctr1,hw_ctr2,hw_ctr3,hw_ctr4,hw_ctr5,hw_ctr6,hw_ctr7,hw_ctr8,hw_ctr9,hw_ctr10,hw_ctr11;
    /// Initializing the Z signal which will have 4*64 dimension
//     float *Z,*Zi;
//     Z = (float *)malloc(N*M*sizeof(float));
//     Zi = (float *)malloc(N*M*sizeof(float));
// 
//     /// Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted by S
//     float *S,*Si;
//     S = (float *)malloc(N*M*sizeof(float));
//     Si = (float *)malloc(N*M*sizeof(float));
// 
//     //\now defining the argument files which will contain the corresponding values of Z and S
//     FILE *Zreal ,*Zimag , *Sreal,*Simag;
//     Zreal=fopen(ZIN,"r");
//     Zimag=fopen(ZIMAGIN,"r");
//     Sreal=fopen(SIN,"r");
//     Simag=fopen(SIMAGIN,"r");
// 
// 
//     //\now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
// 
//     for (int i=0 ; i<N ; i++)
//     {
//         for (int j=0;j<M ; j++)
//         {
//             fscanf(Zreal,"%f",&Z[i*M +j]);
//             fscanf(Zimag,"%f",&Zi[i*M +j]);
//             fscanf(Sreal,"%f",&S[i*M +j]);
//             fscanf(Simag,"%f",&Si[i*M +j]);
//         }
//     }
// 
//     cout<<"Done reading the files from the input provided \n";

    //// Computing the hermitian of S
 //   printf("\n Entered the temporal mitigation ... \n ");
    float *Shermitian,*Shermitianimag;
    Shermitian = (float *)malloc(M*N*sizeof(float));
    Shermitianimag = (float *)malloc(M*N*sizeof(float));




//     cout<<"Calling a function to compute the hermitian \n";
//     hw_ctr.start();
    hermitian(S,Si,Shermitian,Shermitianimag);
//     hw_ctr.stop();
//     uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running hermitian in hardware: "
//         << hw_cycles << std::endl;

    //	float Stemp[64][4],Simagtemp[64][4];
    //
    //		for(int i=0;i<64;i++)
    //			for(int j=0;j<4;j++)
    //			{
    //				Stemp[i][j]=Shermitian[i*4+j];
    //				Simagtemp[i][j]=Shermitianimag[i*4+j];
    //			}

//     cout<<"Printing out the hermitian real part \n";
//     display(Shermitian,64,4);
// 
//     cout<<"Printing out the hermitian imag part \n";
//     display(Shermitianimag,64,4);






    //// Now computing the result from the first multiplication (Z*S^H)--> Result 1
    float *result1,*result1imag;
    result1 = (float *)malloc(N*N*sizeof(float));
    result1imag = (float *)malloc(N*N*sizeof(float));

//     cout<<"Now computing the matrix multiplication of Z*S^H ---> C3 \n ";
//     hw_ctr1.start();
    mmult(Z,Zi,Shermitian,Shermitianimag,result1,result1imag);
//     hw_ctr1.stop();
//     hw_cycles = hw_ctr1.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running mmult in hardware: "
//         << hw_cycles << std::endl;

    //	float result1temp[4][4],result1tempimag[4][4];
    //
    //
    //	for(int i=0;i<4;i++)
    //				for(int j=0;j<4;j++)
    //				{
    //					result1temp[i][j]=result1[i*4+j];
    //					result1tempimag[i][j]=result1imag[i*4+j];
    //				}
    //
    //	cout<<"Real Part \n ";
    //	display(result1,4,4);

//     cout<<"Imag Part \n";
//     display(result1imag,4,4);


    //// Now computing the second matrix multiplication (S*S^H) ---> Result2
    float *result2,*result2imag;
    result2 = (float *)malloc(N*N*sizeof(float));
    result2imag = (float *)malloc(N*N*sizeof(float));


    //	cout<<"Now computing the result of S*S^H ---> C1 \n";
//     hw_ctr2.start();
    mmult(S,Si,Shermitian,Shermitianimag,result2,result2imag);
//     hw_ctr2.stop();
//     hw_cycles = hw_ctr2.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running mmult in hardware: "
//         << hw_cycles << std::endl;


    //	float result2temp[4][4],result2tempimag[4][4];
    //
    //	for(int i=0;i<4;i++)
    //				for(int j=0;j<4;j++)
    //				{
    //					result2temp[i][j]=result2[i*4+j];
    //					result2tempimag[i][j]=result2imag[i*4+j];
    //				}


//     cout<<" ************************** Line 119 check  S*S^H *************** \n";
// 
// 
//     cout<<"Real Part \n";
//     display(result2,4,4);
// 
//     cout<<"Imag part \n";
//     display(result2imag,4,4);
//     //
// 
// 
// 
// 
//     cout<<"\n the result is fine uptil here \n";
    //// Now computing the inverse of the above result which is (S*S^H)^-1 ---> D and Di

    float *inv1,*inv2;
    //	*intmedt1,*intmedt2; // To store inverse of A[][]
    inv1=(float *)malloc(N*N*sizeof(float));
    inv2=(float *)malloc(N*N*sizeof(float));
    //	intmedt1=(float *)malloc(N*N*sizeof(float));
    //	intmedt2=(float *)malloc(N*N*sizeof(float));
// 
//     cout<<"Now computing the inverse of the above matrix multiplication ---> C2=(C1)^-1 \n";
// 
//     //	inverse(result2,inv1);
//     hw_ctr3.start();
    alternateinverse(result2,inv1);
//     hw_ctr3.stop();
//     hw_cycles = hw_ctr3.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running inverse in hardware: "
//         << hw_cycles << std::endl;
// 
// 
// 
//     display(inv1,4,4);
//     cout<<"\n";
// 
//     hw_ctr4.start();
    alternateinverse(result2imag,inv2);
//     hw_ctr4.stop();
//     hw_cycles = hw_ctr4.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running inverse in hardware: "
//         << hw_cycles << std::endl;
// 
//     //	display(inv2,4,4);
// 
// 
//     cout<<"*****Check for line 162 ********\n ";
// 
// 
//     cout<<"Printing the inverse of (S*S^H) C2 --> \n ";

    float *resultreal,*resultimag;
    float *resultrealinv,*resultimaginv;
    resultreal=(float *)malloc(N*N*sizeof(float));
    resultimag=(float *)malloc(N*N*sizeof(float));
    resultrealinv=(float *)malloc(N*N*sizeof(float));
    resultimaginv=(float *)malloc(N*N*sizeof(float));

//     hw_ctr8.start();
    realpart(result2,result2imag,inv1,inv2,resultreal);
    alternateinverse(resultreal,resultrealinv);
//     hw_ctr8.stop();
// 
//     hw_cycles = hw_ctr8.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running real part of inverse in hardware: "
//         << hw_cycles << std::endl;
// 
// 
//     display(resultrealinv,4,4);
//     cout<<"\n";
// 
// 
// 
//     cout<<"\n Fine uptill here too \n";

    // Currently only having trouble with imaginary part of inverse

//     hw_ctr9.start();
    imagpart(result2,result2imag,inv1,inv2,resultimag);
    alternateinverse(resultimag,resultimaginv);
//     hw_ctr9.stop();
//     hw_cycles = hw_ctr9.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running imag part of inverse in hardware: "
//         << hw_cycles << std::endl;
// 
// 
//     cout<<"\n";
//     display(resultimaginv,4,4);

    /** No need for this now that intmedt1 and intmedt2 is dynamically allocated
    //	float *D,*Di;
    //	D= (float *)malloc(N * N * sizeof(float));
    //	Di= (float *)malloc(N * N * sizeof(float));
    //
    //	cout<<"Now copying the above data in a dynamic array D and Di \n";
    //
    //	for (int i=0 ; i<N ; i++)
    //	        {
    //	        	for (int j=0;j<N ; j++)
    //	        	{ D[i*N + j]=intmedt1[i][j];
    //	        	  Di[i*N + j]=intmedt2[i][j];
    //
    //	        	}
    //
    //	        }
    */

    /// Now computing the result of (Z*S^H)*(S.S^H)^-1  ---> result3 which is a 4*4 and 4*4 multiplication

    float *result3,*result3imag;
    result3 = (float *)malloc(N*N*sizeof(float));
    result3imag = (float *)malloc(N*N*sizeof(float));


//     cout<<"Now computing the result (Z*S^H)*(S*S^H)^-1 \n ";
//     hw_ctr5.start();
    mmult4(result1,result1imag,resultrealinv,resultimaginv,result3,result3imag);
//     hw_ctr5.stop();
//     hw_cycles = hw_ctr5.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running hermitian in hardware: "
//         << hw_cycles << std::endl;

    //	float result3temp[4][4],result3imagtemp[4][4];
    //
    //	for (int i=0 ; i<N ; i++)
    //		        {
    //		        	for (int j=0;j<N ; j++)
    //		        	{ result3temp[i][j]=result3[i*N + j];
    //		        	  result3imagtemp[i][j]=result3imag[i*N + j];
    //
    //		        	}
    //
    //		        }

//     cout<<"\n****** Result from line 191 **************\n";
// 
//     cout<<"Result from C3*C2 \n";
//     cout<<"Real Part \n";
//     display(result3,4,4);
// 
//     cout<<"Imag Part \n";
//     display(result3imag,4,4);


    /// Now computing the final matrix multiplication which is result3*S ---> result4 this is 4*4 and 4*64 multiplication

    float *result4,*result4imag;
    result4 = (float *)malloc(N*M*sizeof(float));
    result4imag = (float *)malloc(N*M*sizeof(float));



//     cout<<"Final multiplication is being computed \n";
//     hw_ctr6.start();
    mmult64(result3,result3imag,S,Si,result4,result4imag);
//     hw_ctr6.stop();
//     hw_cycles = hw_ctr6.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running hermitian in hardware: "
//         << hw_cycles << std::endl;


    //	float result4temp[4][64],result4tempimag[4][64];
    //
    //	for (int i=0 ; i<4 ; i++)
    //		        {
    //		        	for (int j=0;j<64 ; j++)
    //		        	{ result4temp[i][j]=result4[i*64+j];
    //		        	  result4tempimag[i][j]=result4imag[i*64+j];
    //
    //		        	}
    //
    //		        }
    //


//     cout<<"*******Final Multiplication result Line 221 *********** \n";
//     cout<<"Real Part \n";
//     display(result4,4,64);
// 
//     cout<<"Imag Part \n";
//     display(result4imag,4,64);
    /// Now we have to compute the final operation which is matrix subtraction : (Z - result4) ---> Zr  4*64 - 4*64
//     float *zres , *zresimag;
//     zres = (float *)malloc(N*M*sizeof(float));
//     zresimag = (float *)malloc(N*M*sizeof(float));



//     cout<<"Now computing the matrix subtraction from the above result to compute the Z response \n";
//     hw_ctr7.start();
    msub(Z,Zi,result4,result4imag,zres,zresimag);
//     hw_ctr7.stop();

//     hw_cycles = hw_ctr7.avg_cpu_cycles();
//     std::cout << "Average number of CPU cycles running hermitian in hardware: "
//         << hw_cycles << std::endl;



//     cout<<"*****Final result being printed ******\n";
//     //// Printing the result out
//     cout<<"Real part: \n";
//     display(zres,4,64);
//     cout<<"Imag part: \n";
//     display(zresimag,4,64);

    //	cout<<"Printing the real part of the result first \n ";
    //	for(int i=0;i<N;i++)
    //	{for(int j=0;j<M;j++)
    //		{ cout<<zres[i*M + j]<<" ";
    //		}
    //		cout<<"\n";
    //
    //	}
    //
    //	cout<<"Printing the imaginary part of the result now \n ";
    //		for(int i=0;i<N;i++)
    //		{for(int j=0;j<M;j++)
    //			{ cout<<zresimag[i*M + j]<<" ";
    //			}
    //			cout<<"\n";
    //
    //		}
    // *** From previous inverse code
    /*
       float *A, *B,*D,*Di;
       A= (float *)malloc(N * N * sizeof(float));
       B= (float *)malloc(N * N * sizeof(float));
       D= (float *)malloc(N * N * sizeof(float));
       Di= (float *)malloc(N * N * sizeof(float));
    //    C_sw= (float *)malloc(N * N * sizeof(float));
    //    C_swi= (float *)malloc(N * N * sizeof(float));

    if (!A || !B || !D || !Di) {
    if (A) free(A);
    //if (Ai) free(Ai);
    if (B) free(B);
    //if (Bi) free(Bi);
    if (D) free(D);
    if (Di) free(Di);
    //              if (C_sw) free(C_sw);
    //              if (C_swi) free(C_swi);
    return 2;
    }
    FILE *data1 , *data2;

    data1=fopen(argv[1],"r");
    data2=fopen(argv[2],"r");

    for (int i=0 ; i<N ; i++)
    {
    for (int j=0;j<N ; j++)
    {
    fscanf(data1,"%f",&A[i*N +j]);
    fscanf(data2,"%f",&B[i*N +j]);
    }
    }

    //    display(A);
    //    display(B);

    float inv1[N][N],inv2[N][N],intmedt1[N][N],intmedt2[N][N]; // To store inverse of A[][]


    cout << "\nThe Inverse is :\n";
    inverse(A, inv1);
    display(inv1);
    inverse(B, inv2);
    display(inv2);
    realpart(A,B,inv1,inv2,intmedt1);
    display(intmedt1);
    imagpart(A,B,inv1,inv2,intmedt2);
    display(intmedt2);


    for (int i=0 ; i<N ; i++)
    {
    for (int j=0;j<N ; j++)
    { D[i*N + j]=intmedt1[i][j];
    Di[i*N + j]=intmedt2[i][j];

    }

    }

    free(A);
    free(B);
    free(D);
    free(Di);
    //    free(C_sw);
    //    free(C_swi);



*/

    ///****
    /// Now freeing up the variables used
//     free(Z);
//     free(Zi);
//     free(S);
//     free(Si);
    free(Shermitian);
    free(Shermitianimag);
    free(result1);
    free(result1imag);
    free(result2);
    free(result2imag);
    free(inv1);
    free(inv2);
    free(resultreal);
    free(resultrealinv);
    free(resultimag);
    free(resultimaginv);
    free(result3);
    free(result3imag);
    free(result4);
    free(result4imag);
//     free(zres);
//     free(zresimag);





    //return 0;

}





