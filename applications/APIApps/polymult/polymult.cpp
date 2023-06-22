#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include "dash.h"

#error "Polymult needs updates, we don't support both single and double precision FFTs at the same time, and DASH apps are currently assuming float input type"

void poly_mult(double* A, double* B, double* out, int size){
  int i,j;
  for(i=0;i<size;i++){
    for(j=0;j<size;j++){
      out[(i + j)*2] += A[i*2] * B[j*2] - A[i*2+1] * B[j*2+1];
      out[(i + j)*2+1] += A[i*2] * B[j*2+1] + A[i*2+1] * B[j*2];
    }
  }
}

int main(void) {
  printf("Starting execution of the non-kernel thread [nk]\n");

  const size_t size = 256;

  double* input = (double*) malloc(4 * size * sizeof(double));
  double* fft_out = (double*) malloc(4 * size * sizeof(double));

  double* input2 = (double*) malloc(4 * size * sizeof(double));
  double* fft_out2 = (double*) malloc(4 * size * sizeof(double));
  
  double* test_output = (double*) malloc(4 * size * sizeof(double));
  double* poly_mult_out = (double*) malloc(4 * size * sizeof(double));
  double* ifft_out = (double*) malloc(4 * size * sizeof(double));

  for (int i = 0; i < size; i++) {
    input[i*2] = rand()%1000;//1.0;//1.0*i;
    input2[i*2] = rand()%1000;//1.0;//1.0*i;
    input[i*2+1] = 0.0 * i;
    input2[i*2+1] = 0.0 * i;
    test_output[i*2] = 0.0;
    test_output[i*2+1] = 0.0;
    poly_mult_out[i*2] = 0.0;
    poly_mult_out[i*2+1] = 0.0;
  }

  for (int i = 0; i < size*2; i++){
	input[size*2+i] = 0.0;
	input2[size*2+i] = 0.0;
	test_output[size*2+i] = 0.0;
    	poly_mult_out[size*2+i] = 0.0;
  }

  poly_mult(input, input2, test_output, size);

  // Poly mult using double precision FFT (FFT followed by IFFT)
  DASH_FFT_P(input, fft_out, size*2, true, DOUBLE_P);
  DASH_FFT_P(input2, fft_out2, size*2, true, DOUBLE_P);
  //DASH_FFT(input, fft_out, size*2, true);
  //DASH_FFT(input2, fft_out2, size*2, true);
  for (int i = 0; i < size*2; i++){
    poly_mult_out[i*2] = fft_out[i*2] * fft_out2[i*2] - fft_out[i*2+1] * fft_out2[i*2+1];
    poly_mult_out[i*2+1] = fft_out[i*2] * fft_out2[i*2+1] + fft_out[i*2+1] * fft_out2[i*2];
  }
  DASH_FFT_P(poly_mult_out, ifft_out, size*2, false, DOUBLE_P);
  //DASH_FFT(poly_mult_out, ifft_out, size*2, false);

  int flag = 0;
  int max_diff = 0;
  int diff = 0;
  for (int i = 0; i < size*2; i++) {
    int index = i*2;//size*4-i*2;
    diff = abs(double(int(ifft_out[index])) - test_output[i*2]);
    if( abs(double(int(ifft_out[index])) - test_output[i*2]) > 1){
      printf("[Poly Mult] Error at index %d: FFT_r:%f PolyMult_r:%f\n", i, double(int(ifft_out[index])), test_output[i*2]);
      flag = 1;
    }
    if(diff>max_diff)
      max_diff = diff;
    //break;
  }

  if(!flag)
    printf("[Poly Mult] Polynomial multiplication operation completed successfully!\n");
  //else
    printf("[Poly Mult] Max diff is %d when using double precision FFT\n", max_diff);

  // Poly mult using single precision FFT (FFT followed by IFFT)
  #error "Polymult needs updates, we don't support both single and double precision FFTs at the same time, and DASH apps are currently assuming float input type"
  DASH_FFT_P(input, fft_out, size*2, true, FLOAT_P);
  DASH_FFT_P(input2, fft_out2, size*2, true, FLOAT_P);
  //DASH_FFT(input, fft_out, size*2, true);
  //DASH_FFT(input2, fft_out2, size*2, true);
  for (int i = 0; i < size*2; i++){
    poly_mult_out[i*2] = fft_out[i*2] * fft_out2[i*2] - fft_out[i*2+1] * fft_out2[i*2+1];
    poly_mult_out[i*2+1] = fft_out[i*2] * fft_out2[i*2+1] + fft_out[i*2+1] * fft_out2[i*2];
  }
  DASH_FFT_P(poly_mult_out, ifft_out, size*2, false, FLOAT_P);
  //DASH_FFT(poly_mult_out, ifft_out, size*2, false);

  flag = 0;
  max_diff = 0;
  diff = 0;
  for (int i = 0; i < size*2; i++) {
    int index = i*2;//size*4-i*2;
    diff = abs(double(int(ifft_out[index])) - test_output[i*2]);
    if( abs(double(int(ifft_out[index])) - test_output[i*2]) > 50){
      printf("[Poly Mult] Error at index %d: FFT_r:%f PolyMult_r:%f\n", i, double(int(ifft_out[index])), test_output[i*2]);
      flag = 1;
    }
    if(diff>max_diff)
      max_diff = diff;
    //break;
  }

  if(!flag)
    printf("[Poly Mult] Polynomial multiplication operation completed successfully!\n");
  //else
    printf("[Poly Mult] Max diff is %d when using single precision FFT\n", max_diff);

  free(input);
  free(fft_out);
  free(input2);
  free(fft_out2);
  free(test_output);
  free(poly_mult_out);
  free(ifft_out);
  printf("[nk] Non-kernel thread execution is complete...\n\n");
  return 0;
}
