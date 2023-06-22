// #include <mex.h>
//#include <math.h>
// #include <matrix.h>
//#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  size_t len, x_count, y_count, i, j, k;
  size_t n_samples; 
  size_t dft_size;

  double lag;
  double T = 0.000512;
  double B = 500000;
  double sampling_rate = 1000;
  double max_corr = 0;
  double index = 0;

  double *c, *d, *X1, *X2, *corr_freq;
  double *time, *received, *dftMatrix, *indftMatrix, *corr, *gen_wave;

  clock_t begin, end;

  FILE *fp;

  int row, column, row2, column2;

  for (i = 0; i < 5; i++) {index = i * 1.1;}
    
  n_samples = 256;
  dft_size = 2 * n_samples - 1;

  begin = clock();
    
  time = malloc(n_samples*sizeof(double));
  received = malloc(2*n_samples*sizeof(double));
  dftMatrix = malloc(2* dft_size * dft_size * sizeof(double));
  indftMatrix = malloc(2 * dft_size * dft_size * sizeof(double));
  corr = malloc( (2*(2*n_samples - 1)) * sizeof(double));
  gen_wave = malloc(2 * n_samples * sizeof(double));

  fp = fopen("./input/time_input.txt","r");
  if (fp == NULL) { printf("Unable to open time_input.txt!\n"); }
  for(i = 0; i < n_samples; i++) {
    fscanf(fp, "%lf", &time[i]);
  }
  fclose(fp);
  end = clock();
  printf("Reading time input: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  for (i = 0; i < 2 * n_samples; i += 2) {
    gen_wave[i] = sin(M_PI * B / T * pow(time[i / 2], 2));
    gen_wave[i + 1] = cos(M_PI * B / T * pow(time[i / 2], 2));
  }
  end = clock();
  printf("Generating waveform: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  fp = fopen("./input/received_input.txt","r");
  if (fp == NULL) { printf("Unable to open received_input.txt!\n"); }
  for(i=0; i<2*n_samples; i++) {
    fscanf(fp,"%lf", &received[i]);
  }
  fclose(fp);
  end = clock();
  printf("Reading received input: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  fp = fopen("./input/dftcoe.txt", "r");
  if (fp == NULL) { printf("Unable to open dftcoe.txt!\n"); }
  for (i = 0; i < dft_size * dft_size *2; i++) {
    fscanf(fp, "%lf", &dftMatrix[i]);
  }
  fclose(fp);
  end = clock();
  printf("Reading DFT Matrix: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  fp = fopen("./input/indftcoe.txt", "r");
  if (fp == NULL) { printf("Unable to open indftcoe.txt!\n"); }
  for (i = 0; i<dft_size * dft_size * 2; i++) {
    fscanf(fp, "%lf", &indftMatrix[i]);
  }
  fclose(fp);
  end = clock();
  printf("Reading IDFT Matrix: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  //Add code for zero-padding, to make sure signals are of same length
  begin = clock();
  len = 2 * n_samples - 1;

  c = malloc(2 * len * sizeof(double));
  d = malloc(2 * len * sizeof(double));

  x_count = 0;
  y_count = 0;

  for (i = 0; i < 2 * len; i += 2) {
    if (i/2 > n_samples - 1) {
      c[i] = gen_wave[x_count]; 
      c[i + 1] = gen_wave[x_count + 1];
      x_count += 2;
    } else {
      c[i] = 0;
      c[i + 1] = 0;
    }
    if (i > n_samples) {
      d[i] = 0;
      d[i + 1] = 0;
    } else {
      d[i] = received[y_count];
      d[i + 1] = received[y_count + 1];
      y_count += 2;
    }
  }
  end = clock();
  printf("Zero padding: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  X1 = malloc(2 * len * sizeof(double));
  X2 = malloc(2 * len * sizeof(double));
  corr_freq = malloc(2 * len * sizeof(double));
  for (i = 0; i < dft_size * dft_size *2; i += 2) {
    row = i /512;
    column = i % 512;
    X1[2*row] += dftMatrix[i] * c[column];
    X1[2*row+1] += dftMatrix[i+1] * c[column+1];
  }
  end = clock();
  printf("DFT 1: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  for (j = 0; j < dft_size * dft_size * 2; j += 2) {
    row2 = j / 512;
    column2 = j % 512;
    X2[2 * row2] += dftMatrix[j] * d[column2];
    X2[2 * row2 + 1] += dftMatrix[j + 1] * d[column2 + 1];
  }
  end = clock();
  printf("DFT 2: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  for (i = 0; i < 2 * len; i += 2) {
    corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
    corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
  }
  end = clock();
  printf("Freq Domain Mult: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  for (i = 0; i < dft_size * dft_size * 2; i += 2) {
    row = i / 512;
    column = i % 512;
    corr[2 * row] += indftMatrix[i] * corr_freq[column];
    corr[2 * row + 1] += indftMatrix[i + 1] * corr_freq[column + 1];
  }
  end = clock();
  printf("Inverse DFT: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);

  begin = clock();
  for (i = 0; i < 2 * (2 * n_samples - 1); i += 2) {
    // Only finding maximum of real part of correlation
    if (corr[i] > max_corr) {
      max_corr = corr[i];
      index = i / 2;
    }
  }
  end = clock();
  printf("Maximum Calculation: %f\n", ((double)(end - begin))/CLOCKS_PER_SEC);
  
  lag = (n_samples - index) / sampling_rate;
  printf("Lag Value is: %lf\n", lag);
}

