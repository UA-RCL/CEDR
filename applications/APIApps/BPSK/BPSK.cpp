#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dash.h"

#error "BPSK needs updates, the BPSK API has been changed"

double randRange()
{
  double min = -5.0;
  double max = 5.0;
  return min + (double) ((double)rand() / (double) (RAND_MAX) * (max - min + 1));
}

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  const size_t nsymbols = 100000;
  const size_t number_snrs = 1001;

  double* input = (double*) malloc(2 * nsymbols * sizeof(double));
  double* output = (double*) malloc(nsymbols * sizeof(double));
  double error;

  double* decisions = (double*) malloc(nsymbols * sizeof(double));
  //double* input_r = (double*) malloc(nsymbols * sizeof(double));
  double distance1_r,distance1_i;
  double distance2_r,distance2_i;
  double dist1,dist2;
  int error_count = 0;
  double error_estimate;


  double* random1 = (double*) malloc(nsymbols * sizeof(double));
  double* random2 = (double*) malloc(nsymbols * sizeof(double));

  FILE *fp;
  fp = fopen("input/random1.txt", "r");
  for (int i = 0; i < nsymbols; i++) {
    //fscanf(fp, "%lf, %lf", &input_r[i], &input_i[i]);
    fscanf(fp, "%lf", &random1[i]);
  }
  fclose(fp);
  fp = fopen("input/random2.txt", "r");
  for (int i = 0; i < nsymbols; i++) {
    //fscanf(fp, "%lf, %lf", &input_r[i], &input_i[i]);
    fscanf(fp, "%lf", &random2[i]);
  }
  fclose(fp);

  fp = fopen("output.txt", "w");

  const double const1_r = 1.0, const1_i = 0.0;
  const double const2_r = -1.0, const2_i = 0.0;

  double snr_now, ebno, sigma;

  srand(time(0));

  for (int snr = 0; snr<number_snrs; snr++){
    snr_now = 0.01*snr;
    ebno=pow(10,(snr_now/10));
    sigma=sqrt(1.0/(ebno));

    for (int i = 0; i < nsymbols; i++){
      // Init decisions to 0
      decisions[i] = 0;
      // Create input
      input[2*i] = -1.0 + sigma*random1[i];//randRange();//((double)rand() / (double)RAND_MAX);
      input[2*i+1] = sigma*random2[i];//randRange();//((double)rand() / (double)RAND_MAX);
    }
    DASH_BPSK(input, decisions, nsymbols);
    error_count = 0;
    for (int i = 0; i < nsymbols; i++){
      if(decisions[i]==1)
  	error_count++;
    }

    error_estimate = (double)error_count / (double)nsymbols;

    //fscanf(fp, "%lf, %lf", &input_r[i], &input_i[i]);
    fprintf(fp, "%1.10lf\n", error_estimate);
  }
  

  fclose(fp);

  free(input);
  free(decisions);
  free(output);
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
