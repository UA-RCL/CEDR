#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "dash.h"

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  const size_t nsymbols = 100000;

  double* input_r = (double*) malloc(nsymbols * sizeof(double));
  double* input_i = (double*) malloc(nsymbols * sizeof(double));
  double* output = (double*) malloc(nsymbols * sizeof(double));
  double error;

  double* decisions = (double*) malloc(nsymbols * sizeof(double));
  //double* input_r = (double*) malloc(nsymbols * sizeof(double));
  double distance1_r,distance1_i;
  double distance2_r,distance2_i;
  double dist1,dist2;
  int error_count = 0;
  double error_estimate;

  const double const1_r = 1.0, const1_i = 0.0;
  const double const2_r = -1.0, const2_i = 0.0;

  FILE *fp;
  fp = fopen("input1.txt", "r");
  for (int i = 0; i < nsymbols; i++) {
    fscanf(fp, "%lf, %lf", &input_r[i], &input_i[i]);
  }
  fclose(fp);

  fp = fopen("output1.txt", "r");
  fscanf(fp, "%lf", &error);
  for (int i = 0; i < nsymbols; i++) {
    fscanf(fp, "%lf", &output[i]);
  }
  fclose(fp);

  // Init decisions to 0
  for (int i = 0; i < nsymbols; i++){
    decisions[i] = 0;
  }

  for (int i = 0; i < nsymbols; i++){
    distance1_r = input_r[i] - const1_r;
    distance1_i = input_i[i] - const1_i;
    distance2_r = input_r[i] - const2_r;
    distance2_i = input_i[i] - const2_i;
    dist1 = sqrt(distance1_r*distance1_r + distance1_i*distance1_i);
    dist2 = sqrt(distance2_r*distance2_r + distance2_i*distance2_i);
    if (dist1<dist2){
      decisions[i] = 1;
      error_count++;
    }
    else{
      decisions[i] = -1;
    }
  }
  error_estimate = (double)error_count / (double)nsymbols;

  printf("[BPSK] Error:%lf\tError Estimate:%lf\n", error, error_estimate);
  for (int i = 0; i < nsymbols; i++) {
    if(decisions[i]!=output[i]){
      distance1_r = input_r[i] - const1_r;
      distance1_i = input_i[i] - const1_i;
      distance2_r = input_r[i] - const2_r;
      distance2_i = input_i[i] - const2_i;
      dist1 = sqrt(distance1_r*distance1_r + distance1_i*distance1_i);
      dist2 = sqrt(distance2_r*distance2_r + distance2_i*distance2_i);
      printf("[BPSK] %d: decision:%lf output:%lf\n", i, decisions[i], output[i]);
      printf("[BPSK] %d: input_r:%lf input_i:%lf\n", i, input_r[i], input_i[i]);
      printf("[BPSK] %d: dist1_r:%lf dist1_i:%lf\n", i, distance1_r, distance1_i);
      printf("[BPSK] %d: dist2_r:%lf dist2_i:%lf\n", i, distance2_r, distance2_i);
      printf("[BPSK] %d: dist1:%lf dist2:%lf\n", i, dist1, dist2);
    }
  }

  free(input_r);
  free(input_i);
  free(output);
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
