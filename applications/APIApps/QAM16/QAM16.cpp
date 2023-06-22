#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dash.h"

#define MAX_SNR                    10
#define SNR_CHANGE                 0.1
#define SNR_SIZE                   (int)(MAX_SNR / SNR_CHANGE)

#define NBITS                      4
#define NINPUTS                    1000

#define CONST_SIZE         NBITS * NBITS

#define RAND_RANGE                 5

int grey[] = { 0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10 };

double randRange() {
  return -1 * RAND_RANGE + (double) ((double)rand() / (double) (RAND_MAX) * (RAND_RANGE - RAND_RANGE + 1));
}

int getBitError(int value1, int value2) {
  int count = 0;
  int n = value1 ^ value2;
   
  while (n) {
    count += n & 1;
    n >>= 1;
  }

  return count;
}

double sum(int* arr, int size) {
  double sum = 0;

  for (int i = 0; i < size; i++) {
    sum += (double) arr[i];
  }

  return sum;
}

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  // Function part
  int* inputs = (int*) calloc(NINPUTS, sizeof(int));

  FILE *input_file = fopen("input/input.txt", "r");

  int line;
  int i = 0;

  printf("Reading from input.txt...\n");

  while (fscanf(input_file, "%d", &line) != EOF) {
    inputs[i] = line;
    i++;
  }

  printf("Done reading from inputs.txt!\n");

  fclose(input_file);
    
  /*for (int i = 0; i < NINPUTS; i++) {
    inputs[i] = rand() % (2^NBITS - 1);
  }*/

  int* inputs_grey = (int*) calloc(NINPUTS, sizeof(int));

  for (int i = 0; i < NINPUTS; i++) {
    inputs_grey[i] = grey[inputs[i]];
  }

  double constReal[] = {-0.948683, -0.948683, -0.948683, -0.948683, -0.316228, -0.316228, -0.316228, -0.316228, 0.316228, 0.316228, 0.316228, 0.316228, 0.948683, 0.948683, 0.948683, 0.948683};
  double constImag[] = {-0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683, -0.948683, -0.316228, 0.316228, 0.948683};

  double* bincReal = (double*) calloc(NINPUTS, sizeof(double));
  double* bincImag = (double*) calloc(NINPUTS, sizeof(double));

  for (int i = 0; i < NINPUTS; i++) {
    bincReal[i] = constReal[inputs[i]];
    bincImag[i] = constImag[inputs[i]];
  }

  int* decisions_bin = (int*) calloc(NINPUTS, sizeof(int));
  int* decisions_bin_grey = (int*) calloc(NINPUTS, sizeof(int));
  double* berr_estimate = (double*) calloc((SNR_SIZE + 1), sizeof(double));
  double* berr_estimate_grey = (double*) calloc((SNR_SIZE + 1), sizeof(double));
  double* input = (double*) calloc(NINPUTS * 2, sizeof(double));
  int* error = (int*) calloc(NINPUTS, sizeof(int));
  int* error_grey = (int*) calloc(NINPUTS, sizeof(int));
  
  FILE *randreal_file = fopen("input/randreal.txt", "r");
  FILE *randimag_file = fopen("input/randimag.txt", "r");

  FILE *decisions_file = fopen("decResults.txt", "w+");

  int k = 0;
    
    double linereal;
    double lineimag;
    int fCountReal = 0;
    int fCountImag = 0;
    
    double randreal[NINPUTS*100];
    double randimag[NINPUTS*100];
    
    //printf("Reading from randreal.txt...\n");

    while(fCountReal < NINPUTS*100) {
      fscanf(randreal_file, "%lf", &linereal);
      //printf("FCountReal: %d\n", fCountReal);
      randreal[fCountReal] = linereal;
      fCountReal++;
    }

    //printf("Done reading from randreal.txt!\n");

    //printf("Reading from randimag.txt\n");

    while(fCountImag < NINPUTS*100) {
      fscanf(randimag_file, "%lf", &lineimag);
      randimag[fCountImag] = lineimag;
      fCountImag++;
    }
  fclose(randimag_file);
  fclose(randreal_file);

    //printf("Done reading from randimag.txt!\n");

  for (double snr = 0; snr <= MAX_SNR; snr += SNR_CHANGE) {
    double variance = sqrt(1/pow(10, (snr/10)));
    for (int i = 0; i < NINPUTS; i++) {
      input[2 * i] = bincReal[i] + (variance * randreal[i+(int)(snr*10)])/sqrt(10); // constReal[(*input)[2*i]];
      input[2 * i + 1] = bincImag[i] + (variance * randimag[i+(int)(snr*10)])/sqrt(10); // constImag[(*input)[2*i]];
    }

    // Call DASH_QAM16
    // lol scope
    {
      dash_cmplx_flt_type *qam_inp = (dash_cmplx_flt_type*) malloc(NINPUTS * sizeof(dash_cmplx_flt_type));

      for (size_t i = 0; i < NINPUTS; i++) {
        qam_inp[i].re = (dash_re_flt_type) input[2*i];
        qam_inp[i].im = (dash_re_flt_type) input[2*i+1];
      }

      DASH_QAM16_flt(qam_inp, decisions_bin, NINPUTS);

      free(qam_inp);
    }
    
/*  
* IO Read Write for plots
    if (snr == 0){
      FILE *fp1,*fp2;
      fp1 = fopen("QAM16Inputs.txt", "w");
      fp2 = fopen("QAM16Decisions.txt", "w");
      for (int i = 0; i < NINPUTS; i++){
        fprintf(fp1, "%lf %lf\n", input[2*i], input[2*i+1]);
        fprintf(fp2, "%d\n", decisions_bin[i]);
      }
      fclose(fp1);
      fclose(fp2);
    }
*/

    for (int i = 0; i < NINPUTS; i++) {
      decisions_bin_grey[i] = grey[decisions_bin[i]];
      //fprintf(decisions_file, "%d\n", decisions_bin[i]);
    }
    
    berr_estimate[k] = berr_estimate[k] + (sum(error, NINPUTS)/NINPUTS);
    berr_estimate_grey[k] = berr_estimate_grey[k] + (sum(error_grey, NINPUTS)/NINPUTS);
    k++;
  }
  
  for (int i = 0; i < NINPUTS; i++) {
    error[i] = getBitError(decisions_bin[i], inputs[i]);
    error_grey[i] = getBitError(decisions_bin_grey[i], inputs_grey[i]);
    fprintf(decisions_file, "%d\n", error[i]);
  }

  fclose(decisions_file);
  
  free(error_grey);
  free(error);
  free(berr_estimate_grey);
  free(berr_estimate);
  free(decisions_bin_grey);
  free(decisions_bin);
  free(bincImag);
  free(bincReal);
  free(inputs_grey);
  free(inputs);
  
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
