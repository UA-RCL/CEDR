
#include <iostream>
#include <cstdio>
#include "temporal_mitigation.hpp"
#include <unistd.h>
#include <stdlib.h>
#define PROGPATH "./input/"
#define ZIN PROGPATH "z.txt"
#define ZIMAGIN PROGPATH "zimag.txt"
#define SIN PROGPATH "s.txt"
#define SIMAGIN PROGPATH "simag.txt"
#include <fstream>
#include <stdio.h>
int TM_iter;
float *Z, *Zi;
float *S, *Si;
float *Z_inter_buffer, *Zi_inter_buffer;
float *S_inter_buffer, *Si_inter_buffer;
float *Shermitian, *Shermitianimag;
int M, N;
float *result1, *result1imag;
float *result2, *result2imag;
float *bufferinv4, *bufferinv5;
float *result3, *result3imag;
float *result4, *result4imag;
float *zres, *zresimag;
int frame_it;

float *_Z, *_Zi;
float *_S, *_Si;
float *_Z_inter_buffer, *_Zi_inter_buffer;
float *_S_inter_buffer, *_Si_inter_buffer;
float *_Shermitian, *_Shermitianimag;
float *_result1, *_result1imag;
float *_result2, *_result2imag;
float *_bufferinv4, *_bufferinv5;
float *_result3, *_result3imag;
float *_result4, *_result4imag;
float *_zres, *_zresimag;
void __attribute__((constructor)) setup(void) {
  printf("[Temporal Mitigation] initializing buffers\n");

  TM_iter = 0;
  N = 4;
  M = 64;
  frame_it = 0;
  //FILE *Zreal, *Zimag, *Sreal, *Simag;


  // Initializing the Z signal which will have 4*64 dimension
  Z = (float *)malloc(N * M * sizeof(float));
  Zi = (float *)malloc(N * M * sizeof(float));
  _Z = (float *)malloc(N * M * sizeof(float));
  _Zi = (float *)malloc(N * M * sizeof(float));
  Z_inter_buffer = (float *)malloc(N * M * sizeof(float));
  Zi_inter_buffer = (float *)malloc(N * M * sizeof(float));
  _Z_inter_buffer = (float *)malloc(N * M * sizeof(float));
  _Zi_inter_buffer = (float *)malloc(N * M * sizeof(float));
  // Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted
  // by S
  S = (float *)malloc(N * M * sizeof(float));
  Si = (float *)malloc(N * M * sizeof(float));
  _S = (float *)malloc(N * M * sizeof(float));
  _Si = (float *)malloc(N * M * sizeof(float));
  S_inter_buffer = (float *)malloc(N * M * sizeof(float));
  Si_inter_buffer = (float *)malloc(N * M * sizeof(float));
  _S_inter_buffer = (float *)malloc(N * M * sizeof(float));
  _Si_inter_buffer = (float *)malloc(N * M * sizeof(float));
  // now defining the argument files which will contain the corresponding values of Z and S
  //Zreal = fopen(ZIN, "r");
  //Zimag = fopen(ZIMAGIN, "r");
  //Sreal = fopen(SIN, "r");
  //Simag = fopen(SIMAGIN, "r");

  //// now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
  //for (int i = 0; i < N; i++) {
  //    for (int j = 0; j < M; j++) {
  //        fscanf(Zreal, "%f", &Z[i * M + j]); Z[i * M + j] /= 10.0f;
  //        fscanf(Zimag, "%f", &Zi[i * M + j]); Zi[i * M + j] /= 10.0f;
  //        fscanf(Sreal, "%f", &S[i * M + j]); S[i * M + j] /= 10.0f;
  //        fscanf(Simag, "%f", &Si[i * M + j]); Si[i * M + j] /= 10.0f;
  //    }
  //}
   
  //// Computing the hermitian of S
  Shermitian = (float *)malloc(M * N * sizeof(float));
  Shermitianimag = (float *)malloc(M * N * sizeof(float));
  result1 = (float *)malloc(N * N * sizeof(float));
  result1imag = (float *)malloc(N * N * sizeof(float));

  result2 = (float *)malloc(N * N * sizeof(float));
  result2imag = (float *)malloc(N * N * sizeof(float));
  
  bufferinv4 = (float *)malloc(N * N * sizeof(float));
  bufferinv5 = (float *)malloc(N * N * sizeof(float));
  
  result3 = (float *)malloc(N * N * sizeof(float));
  result3imag = (float *)malloc(N * N * sizeof(float));

  result4 = (float *)malloc(N * M * sizeof(float));
  result4imag = (float *)malloc(N * M * sizeof(float));


  zres = (float *)malloc(N * M * sizeof(float));
  zresimag = (float *)malloc(N * M * sizeof(float));
  
   _Shermitian = (float *)malloc(M * N * sizeof(float));
  _Shermitianimag = (float *)malloc(M * N * sizeof(float));
  _result1 = (float *)malloc(N * N * sizeof(float));
  _result1imag = (float *)malloc(N * N * sizeof(float));

  _result2 = (float *)malloc(N * N * sizeof(float));
  _result2imag = (float *)malloc(N * N * sizeof(float));
  
  _bufferinv4 = (float *)malloc(N * N * sizeof(float));
  _bufferinv5 = (float *)malloc(N * N * sizeof(float));
  
  _result3 = (float *)malloc(N * N * sizeof(float));
  _result3imag = (float *)malloc(N * N * sizeof(float));

  _result4 = (float *)malloc(N * M * sizeof(float));
  _result4imag = (float *)malloc(N * M * sizeof(float));


  _zres = (float *)malloc(N * M * sizeof(float));
  _zresimag = (float *)malloc(N * M * sizeof(float));
  
  printf("[Temporal Mitigation] initialization complete\n");
  remove("cedr_TM_output.txt");
  
}


void __attribute__((destructor)) clean_app(void) {
 printf("[Temporal mitigation] destroying buffers\n");
 free(Z);
 free(Zi);
 free(S);
 free(Si);
 free(Z_inter_buffer);
 free(Zi_inter_buffer);
 free(S_inter_buffer);
 free(Si_inter_buffer);
 free(Shermitian);
 free(Shermitianimag);
 free(result1);
 free(result1imag);
 free(result2);
 free(result2imag);
 free(bufferinv4);
 free(bufferinv5);
 free(result3);
 free(result3imag);

 free(result4);
 free(result4imag);


 free(zres);
 free(zresimag);

  // Virtual Address to DMA Control Slave
 free(_Z);
 free(_Zi);
 free(_S);
 free(_Si);
 free(_Z_inter_buffer);
 free(_Zi_inter_buffer);
 free(_S_inter_buffer);
 free(_Si_inter_buffer);
 free(_Shermitian);
 free(_Shermitianimag);
 free(_result1);
 free(_result1imag);
 free(_result2);
 free(_result2imag);
 free(_bufferinv4);
 free(_bufferinv5);
 free(_result3);
 free(_result3imag);

 free(_result4);
 free(_result4imag);

 free(_zres);
 free(_zresimag);
 
 printf("[Temporal mitigation] buffers destroyed\n");
  
}


extern "C" void TM_head_node (void){
  static bool buffer_select0 = false;
  FILE *Zreal, *Zimag, *Sreal, *Simag;
  FILE *_Zreal, *_Zimag, *_Sreal, *_Simag;
  Zreal = fopen(ZIN, "r");
  Zimag = fopen(ZIMAGIN, "r");
  Sreal = fopen(SIN, "r");
  Simag = fopen(SIMAGIN, "r");
  _Zreal = fopen(ZIN, "r");
  _Zimag = fopen(ZIMAGIN, "r");
  _Sreal = fopen(SIN, "r");
  _Simag = fopen(SIMAGIN, "r");

  // now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
  
  if(buffer_select0){
	  for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			fscanf(Zreal, "%f", &Z[i * M + j]);
				fscanf(Zimag, "%f", &Zi[i * M + j]);
				fscanf(Sreal, "%f", &S[i * M + j]);
				fscanf(Simag, "%f", &Si[i * M + j]);
				Z[i * M + j] =(Z[i * M + j] + TM_iter) / 10.0f;
				Zi[i * M + j] =(Zi[i * M + j] + TM_iter) / 10.0f;
				S[i * M + j] = (S[i * M + j] + TM_iter)/ 10.0f;
				Si[i * M + j]= (Si[i * M + j] + TM_iter)/ 10.0f;
		}
	  }
	  // printf("%lf , %lf ", Z[0], Zi[0]);
  }
  else {
	 for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			fscanf(_Zreal, "%f", &_Z[i * M + j]);
				fscanf(_Zimag, "%f", &_Zi[i * M + j]);
				fscanf(_Sreal, "%f", &_S[i * M + j]);
				fscanf(_Simag, "%f", &_Si[i * M + j]);
				_Z[i * M + j] =(_Z[i * M + j] + TM_iter) / 10.0f;
				_Zi[i * M + j] =(_Zi[i * M + j] + TM_iter) / 10.0f;
				_S[i * M + j] = (_S[i * M + j] + TM_iter)/ 10.0f;
				_Si[i * M + j]= (_Si[i * M + j] + TM_iter)/ 10.0f;
		}
	 }
	 // printf("%lf , %lf ", _Z[0], _Zi[0]);
  }
  printf("\n");
  /*
  if(buffer_select0){
	  printf("\n*******************************************************\n" );
	  printf("Head S array Value\n");
	  printf("Real %f \n", S[0 * N + 0] );
	  printf("Img %f ", Si[0 * N + 0] );
	  for (int i = 0; i < N; i++) {
		  for (int j = 0; j < M; j++) {
			  printf("%f ", S[i * M + j]);
		  }
		  printf("\n");
	  }
	  
	  printf("\n*******************************************************\n" );
  } 
  */
  
   fclose(Zreal);
   fclose(Zimag);
   fclose(Sreal);
   fclose(Simag);
   fclose(_Zreal);
   fclose(_Zimag);
   fclose(_Sreal);
   fclose(_Simag);
	// TM_iter++;
   buffer_select0 = !buffer_select0;

}

extern "C" void TM_hermitian (void){
	static bool buffer_select1 = false;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if(buffer_select1){
			Shermitian[j * N + i] = S[i * M + j];
			Shermitianimag[j * N + i] = -Si[i * M + j];
			}
			else{
			_Shermitian[j * N + i] = _S[i * M + j];
			_Shermitianimag[j * N + i] = -_Si[i * M + j];
			}
		}
	}
	/*
	if(buffer_select1){
		printf("\n*******************************************************\n" );
		printf("Hermitian S array Value\n ");
		printf("Real %f \n", S[0 * N + 0] );
		printf("Img %f ", Si[0 * N + 0] );
		
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				printf("%f ", S[i * N + j] );
				//printf("%f ", Si[i * N + j] );
			}
			printf("\n");
		}
		
		printf("\n*******************************************************\n" );
	}
	*/
	 buffer_select1 = !buffer_select1;
}


extern "C" void TM_Z_buffer (void){
	static bool buffer_select2 = false;
	for (int i = 0; i < N*M; i++) {
		if(buffer_select2){
		Z_inter_buffer[i] = Z[i];		
		Zi_inter_buffer[i] = Zi[i];
		}
		else{
		_Z_inter_buffer[i] = _Z[i];		
		_Zi_inter_buffer[i] = _Zi[i];
		}			
	}
	buffer_select2 = !buffer_select2;
}
extern "C" void TM_S_buffer (void){
	static bool buffer_select3 = false;
	for (int i = 0; i < N*M; i++) {
		if(buffer_select3){
		S_inter_buffer[i] = S[i];		
		Si_inter_buffer[i] = Si[i];		
		}
		else{
		_S_inter_buffer[i] = _S[i];		
		_Si_inter_buffer[i] = _Si[i];		
		}
	}
	buffer_select3 = !buffer_select3;
}

extern "C" void TM_mmult_Z (void){
    static bool buffer_select4 = false;
    float res1 = 0, res2 = 0, res3 = 0, res4 = 0;
    float term1, term2, term3, term4;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res1 = 0; res2 = 0; res3 = 0; res4 = 0;
            for (int k = 0; k < M; k++) {
				if(buffer_select4) {
					//float term1 = Abuf[i][k] * Bbuf[k][j];
					term1 = Z[i * M + k] * Shermitian[k * N + j];
					res1 += term1;
					//float term2 = Aibuf[i][k] * Bibuf[k][j];
					term2 = Zi[i * M + k] * Shermitianimag[k * N + j];
					res2 += term2;
					//float term3 = Abuf[i][k] * Bibuf[k][j];
					term3 = Z[i * M + k] * Shermitianimag[k * N + j];
					res3 += term3;
					//float term4 = Aibuf[i][k] * Bbuf[k][j];
					term4 = Zi[i * M + k] * Shermitian[k * N + j];
					res4 += term4;
				}
				else{
					//float term1 = Abuf[i][k] * Bbuf[k][j];
					term1 = _Z[i * M + k] * _Shermitian[k * N + j];
					res1 += term1;
					//float term2 = Aibuf[i][k] * Bibuf[k][j];
					term2 = _Zi[i * M + k] * _Shermitianimag[k * N + j];
					res2 += term2;
					//float term3 = Abuf[i][k] * Bibuf[k][j];
					term3 = _Z[i * M + k] * _Shermitianimag[k * N + j];
					res3 += term3;
					//float term4 = Aibuf[i][k] * Bbuf[k][j];
					term4 = _Zi[i * M + k] * _Shermitian[k * N + j];
					res4 += term4;
				}
            }
			if(buffer_select4){
				result1[i * N + j] = res1 - res2;
				result1imag[i * N + j] = res3 + res4;
			}
			else{
				_result1[i * N + j] = res1 - res2;
				_result1imag[i * N + j] = res3 + res4;
			}
        }
    }
	printf("\n######################Z Matrix after the MMULT######################\n");
	if(buffer_select4){
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%lf , %lf \n", result1[i * N + j], result1imag[i * N + j]);
			}
		}
		}
	else{
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%lf , %lf \n", _result1[i * N + j], _result1imag[i * N + j]);
			}
		}
	}
	printf("##################################################################\n");
	buffer_select4 = !buffer_select4;
}


extern "C" void TM_mmult_S (void){
	static bool buffer_select5 = false;
    float res1 = 0, res2 = 0, res3 = 0, res4 = 0;
    float term1, term2, term3, term4;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res1 = 0; res2 = 0; res3 = 0; res4 = 0;
            for (int k = 0; k < M; k++) {
				if(buffer_select5){
					//printf("%f  ", S[i * M + k] );
					//float term1 = Abuf[i][k] * Bbuf[k][j];
					term1 = S[i * M + k] * Shermitian[k * N + j];
					res1 += term1;
					//float term2 = Aibuf[i][k] * Bibuf[k][j];
					term2 = Si[i * M + k] * Shermitianimag[k * N + j];
					res2 += term2;
					//float term3 = Abuf[i][k] * Bibuf[k][j];
					term3 = S[i * M + k] * Shermitianimag[k * N + j];
					res3 += term3;
					//float term4 = Aibuf[i][k] * Bbuf[k][j];
					term4 = Si[i * M + k] * Shermitian[k * N + j];
					res4 += term4;
				}
				else {
					//float term1 = Abuf[i][k] * Bbuf[k][j];
					term1 = _S[i * M + k] * _Shermitian[k * N + j];
					res1 += term1;
					//float term2 = Aibuf[i][k] * Bibuf[k][j];
					term2 = _Si[i * M + k] * _Shermitianimag[k * N + j];
					res2 += term2;
					//float term3 = Abuf[i][k] * Bibuf[k][j];
					term3 = _S[i * M + k] * _Shermitianimag[k * N + j];
					res3 += term3;
					//float term4 = Aibuf[i][k] * Bbuf[k][j];
					term4 = _Si[i * M + k] * _Shermitian[k * N + j];
					res4 += term4;
				}
            }
			if(buffer_select5){	
				result2[i * N + j] = res1 - res2;
				result2imag[i * N + j] = res3 + res4;
			}
			else {
				_result2[i * N + j] = res1 - res2;
				_result2imag[i * N + j] = res3 + res4;
			}
		}
    }
	
	/*
	if(buffer_select5){	
		printf("\n*******************************************************\n" );
		printf("Mmmult_S S array value\n");
		printf("Real %f \n ", S[0* M + 0] );
		printf("Img %f ", Si[0 * N + 0] );
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%f ", result2[i * N + j] );
				printf("%f ", result2imag[i * N + j] );
			}
			printf("/n");
		}
		printf("\n*******************************************************\n" );
	}
	*/
	
	buffer_select5 = !buffer_select5;
}


extern "C" void TM_inverse (void){
	static bool buffer_select6 = false;
    float *inv1;
    float *bufferinv1, *bufferinv2, *bufferinv3;
    inv1 = (float *)malloc(N * N * sizeof(float));
    //inv2 = (float *)malloc(N * N * sizeof(float));
    //intmedt1 = (float *)malloc(N * N * sizeof(float));
    //intmedt2 = (float *)malloc(N * N * sizeof(float));
    //intmedb1 = (float *)malloc(N * N * sizeof(float));
    //intmedb2 = (float *)malloc(N * N * sizeof(float));
    //intmedb3 = (float *)malloc(N * N * sizeof(float));
    //intmedb4 = (float *)malloc(N * N * sizeof(float));
    //buffer1 = (float *)malloc(N * N * sizeof(float));
    //buffer2 = (float *)malloc(N * N * sizeof(float));
    //buffer3 = (float *)malloc(N * N * sizeof(float));
    //buffer4 = (float *)malloc(N * N * sizeof(float));
    // The following arrays are used for the inverse computation

    bufferinv1 = (float *)malloc(N * N * sizeof(float));
    bufferinv2 = (float *)malloc(N * N * sizeof(float));
    bufferinv3 = (float *)malloc(N * N * sizeof(float));
    {//alternateinverse(result2, inv1);
	float inv[16], det;
	int i;

	float mcpy[16];
	for (int i = 0; i < 16; i++) {
		if(buffer_select6)
			mcpy[i] = result2[i];
		else
			mcpy[i] = _result2[i];
	}

	inv[0] = mcpy[5] * mcpy[10] * mcpy[15] - mcpy[5] * mcpy[11] * mcpy[14] - mcpy[9] * mcpy[6] * mcpy[15] +
	         mcpy[9] * mcpy[7] * mcpy[14] + mcpy[13] * mcpy[6] * mcpy[11] - mcpy[13] * mcpy[7] * mcpy[10];

	inv[4] = -mcpy[4] * mcpy[10] * mcpy[15] + mcpy[4] * mcpy[11] * mcpy[14] + mcpy[8] * mcpy[6] * mcpy[15] -
	         mcpy[8] * mcpy[7] * mcpy[14] - mcpy[12] * mcpy[6] * mcpy[11] + mcpy[12] * mcpy[7] * mcpy[10];

	inv[8] = mcpy[4] * mcpy[9] * mcpy[15] - mcpy[4] * mcpy[11] * mcpy[13] - mcpy[8] * mcpy[5] * mcpy[15] +
	         mcpy[8] * mcpy[7] * mcpy[13] + mcpy[12] * mcpy[5] * mcpy[11] - mcpy[12] * mcpy[7] * mcpy[9];

	inv[12] = -mcpy[4] * mcpy[9] * mcpy[14] + mcpy[4] * mcpy[10] * mcpy[13] + mcpy[8] * mcpy[5] * mcpy[14] -
	          mcpy[8] * mcpy[6] * mcpy[13] - mcpy[12] * mcpy[5] * mcpy[10] + mcpy[12] * mcpy[6] * mcpy[9];

	inv[1] = -mcpy[1] * mcpy[10] * mcpy[15] + mcpy[1] * mcpy[11] * mcpy[14] + mcpy[9] * mcpy[2] * mcpy[15] -
	         mcpy[9] * mcpy[3] * mcpy[14] - mcpy[13] * mcpy[2] * mcpy[11] + mcpy[13] * mcpy[3] * mcpy[10];

	inv[5] = mcpy[0] * mcpy[10] * mcpy[15] - mcpy[0] * mcpy[11] * mcpy[14] - mcpy[8] * mcpy[2] * mcpy[15] +
	         mcpy[8] * mcpy[3] * mcpy[14] + mcpy[12] * mcpy[2] * mcpy[11] - mcpy[12] * mcpy[3] * mcpy[10];

	inv[9] = -mcpy[0] * mcpy[9] * mcpy[15] + mcpy[0] * mcpy[11] * mcpy[13] + mcpy[8] * mcpy[1] * mcpy[15] -
	         mcpy[8] * mcpy[3] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[11] + mcpy[12] * mcpy[3] * mcpy[9];

	inv[13] = mcpy[0] * mcpy[9] * mcpy[14] - mcpy[0] * mcpy[10] * mcpy[13] - mcpy[8] * mcpy[1] * mcpy[14] +
	          mcpy[8] * mcpy[2] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[10] - mcpy[12] * mcpy[2] * mcpy[9];

	inv[2] = mcpy[1] * mcpy[6] * mcpy[15] - mcpy[1] * mcpy[7] * mcpy[14] - mcpy[5] * mcpy[2] * mcpy[15] +
	         mcpy[5] * mcpy[3] * mcpy[14] + mcpy[13] * mcpy[2] * mcpy[7] - mcpy[13] * mcpy[3] * mcpy[6];

	inv[6] = -mcpy[0] * mcpy[6] * mcpy[15] + mcpy[0] * mcpy[7] * mcpy[14] + mcpy[4] * mcpy[2] * mcpy[15] -
	         mcpy[4] * mcpy[3] * mcpy[14] - mcpy[12] * mcpy[2] * mcpy[7] + mcpy[12] * mcpy[3] * mcpy[6];

	inv[10] = mcpy[0] * mcpy[5] * mcpy[15] - mcpy[0] * mcpy[7] * mcpy[13] - mcpy[4] * mcpy[1] * mcpy[15] +
	          mcpy[4] * mcpy[3] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[7] - mcpy[12] * mcpy[3] * mcpy[5];

	inv[14] = -mcpy[0] * mcpy[5] * mcpy[14] + mcpy[0] * mcpy[6] * mcpy[13] + mcpy[4] * mcpy[1] * mcpy[14] -
	          mcpy[4] * mcpy[2] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[6] + mcpy[12] * mcpy[2] * mcpy[5];

	inv[3] = -mcpy[1] * mcpy[6] * mcpy[11] + mcpy[1] * mcpy[7] * mcpy[10] + mcpy[5] * mcpy[2] * mcpy[11] -
	         mcpy[5] * mcpy[3] * mcpy[10] - mcpy[9] * mcpy[2] * mcpy[7] + mcpy[9] * mcpy[3] * mcpy[6];

	inv[7] = mcpy[0] * mcpy[6] * mcpy[11] - mcpy[0] * mcpy[7] * mcpy[10] - mcpy[4] * mcpy[2] * mcpy[11] +
	         mcpy[4] * mcpy[3] * mcpy[10] + mcpy[8] * mcpy[2] * mcpy[7] - mcpy[8] * mcpy[3] * mcpy[6];

	inv[11] = -mcpy[0] * mcpy[5] * mcpy[11] + mcpy[0] * mcpy[7] * mcpy[9] + mcpy[4] * mcpy[1] * mcpy[11] -
	          mcpy[4] * mcpy[3] * mcpy[9] - mcpy[8] * mcpy[1] * mcpy[7] + mcpy[8] * mcpy[3] * mcpy[5];

	inv[15] = mcpy[0] * mcpy[5] * mcpy[10] - mcpy[0] * mcpy[6] * mcpy[9] - mcpy[4] * mcpy[1] * mcpy[10] +
	          mcpy[4] * mcpy[2] * mcpy[9] + mcpy[8] * mcpy[1] * mcpy[6] - mcpy[8] * mcpy[2] * mcpy[5];

	det = mcpy[0] * inv[0] + mcpy[1] * inv[4] + mcpy[2] * inv[8] + mcpy[3] * inv[12];

	if (det == 0) {
		//std::cout << "/n Breaking out the determinant is 0 /n";
	} else {
		det = 1.0 / det;

		for (int i = 0; i < 16; i++) inv1[i] = inv[i] * det;
	}


    }

    {//mmultiply(inv1, result2imag, bufferinv1);
        int i, j;
        float Abuf[N][N], Bbuf[N][N];

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        Abuf[i][j] = inv1[i * N + j];
						if(buffer_select6)
							Bbuf[i][j] = result2imag[i * N + j];
						else
							Bbuf[i][j] = _result2imag[i * N + j];
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        float result1 = 0.0;
                        for (int k = 0; k < N; k++) {
                                result1 += Abuf[i][k] * Bbuf[k][j];
                        }
                        bufferinv1[i * N + j] = result1;
                }
        }

   



    }

    {// mmultiply(result2imag, bufferinv1, bufferinv2);

        int i, j;
        float Abuf[N][N], Bbuf[N][N];

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
					if(buffer_select6)
                        Abuf[i][j] = result2imag[i * N + j];
					else
						Abuf[i][j] = _result2imag[i * N + j];
                        Bbuf[i][j] = bufferinv1[i * N + j];
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        float l_result1 = 0.0;
                        for (int k = 0; k < N; k++) {
                                l_result1 += Abuf[i][k] * Bbuf[k][j];
                        }
                        bufferinv2[i * N + j] = l_result1;
                }
        }


    }


    {//mmadd(bufferinv2, result2, bufferinv3);
        float Abuf[N][N], Bbuf[N][N];

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        Abuf[i][j] = bufferinv2[i * N + j];
                        //                     Aibuf[i][j] = Ai[i*M + j];
						if(buffer_select6)
							Bbuf[i][j] = result2[i * N + j];
						else
							Bbuf[i][j] = _result2[i * N + j];
                        //                     Bibuf[i][j] = Bi[i*M + j];
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        bufferinv3[i * N + j] = float(Abuf[i][j] + Bbuf[i][j]);
                }
        }

    }

    {//alternateinverse(bufferinv3, bufferinv4);

	float inv[16], det;
	int i;

	float mcpy[16];
	for (int i = 0; i < 16; i++) {
		mcpy[i] = bufferinv3[i];
	}

	inv[0] = mcpy[5] * mcpy[10] * mcpy[15] - mcpy[5] * mcpy[11] * mcpy[14] - mcpy[9] * mcpy[6] * mcpy[15] +
	         mcpy[9] * mcpy[7] * mcpy[14] + mcpy[13] * mcpy[6] * mcpy[11] - mcpy[13] * mcpy[7] * mcpy[10];

	inv[4] = -mcpy[4] * mcpy[10] * mcpy[15] + mcpy[4] * mcpy[11] * mcpy[14] + mcpy[8] * mcpy[6] * mcpy[15] -
	         mcpy[8] * mcpy[7] * mcpy[14] - mcpy[12] * mcpy[6] * mcpy[11] + mcpy[12] * mcpy[7] * mcpy[10];

	inv[8] = mcpy[4] * mcpy[9] * mcpy[15] - mcpy[4] * mcpy[11] * mcpy[13] - mcpy[8] * mcpy[5] * mcpy[15] +
	         mcpy[8] * mcpy[7] * mcpy[13] + mcpy[12] * mcpy[5] * mcpy[11] - mcpy[12] * mcpy[7] * mcpy[9];

	inv[12] = -mcpy[4] * mcpy[9] * mcpy[14] + mcpy[4] * mcpy[10] * mcpy[13] + mcpy[8] * mcpy[5] * mcpy[14] -
	          mcpy[8] * mcpy[6] * mcpy[13] - mcpy[12] * mcpy[5] * mcpy[10] + mcpy[12] * mcpy[6] * mcpy[9];

	inv[1] = -mcpy[1] * mcpy[10] * mcpy[15] + mcpy[1] * mcpy[11] * mcpy[14] + mcpy[9] * mcpy[2] * mcpy[15] -
	         mcpy[9] * mcpy[3] * mcpy[14] - mcpy[13] * mcpy[2] * mcpy[11] + mcpy[13] * mcpy[3] * mcpy[10];

	inv[5] = mcpy[0] * mcpy[10] * mcpy[15] - mcpy[0] * mcpy[11] * mcpy[14] - mcpy[8] * mcpy[2] * mcpy[15] +
	         mcpy[8] * mcpy[3] * mcpy[14] + mcpy[12] * mcpy[2] * mcpy[11] - mcpy[12] * mcpy[3] * mcpy[10];

	inv[9] = -mcpy[0] * mcpy[9] * mcpy[15] + mcpy[0] * mcpy[11] * mcpy[13] + mcpy[8] * mcpy[1] * mcpy[15] -
	         mcpy[8] * mcpy[3] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[11] + mcpy[12] * mcpy[3] * mcpy[9];

	inv[13] = mcpy[0] * mcpy[9] * mcpy[14] - mcpy[0] * mcpy[10] * mcpy[13] - mcpy[8] * mcpy[1] * mcpy[14] +
	          mcpy[8] * mcpy[2] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[10] - mcpy[12] * mcpy[2] * mcpy[9];

	inv[2] = mcpy[1] * mcpy[6] * mcpy[15] - mcpy[1] * mcpy[7] * mcpy[14] - mcpy[5] * mcpy[2] * mcpy[15] +
	         mcpy[5] * mcpy[3] * mcpy[14] + mcpy[13] * mcpy[2] * mcpy[7] - mcpy[13] * mcpy[3] * mcpy[6];

	inv[6] = -mcpy[0] * mcpy[6] * mcpy[15] + mcpy[0] * mcpy[7] * mcpy[14] + mcpy[4] * mcpy[2] * mcpy[15] -
	         mcpy[4] * mcpy[3] * mcpy[14] - mcpy[12] * mcpy[2] * mcpy[7] + mcpy[12] * mcpy[3] * mcpy[6];

	inv[10] = mcpy[0] * mcpy[5] * mcpy[15] - mcpy[0] * mcpy[7] * mcpy[13] - mcpy[4] * mcpy[1] * mcpy[15] +
	          mcpy[4] * mcpy[3] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[7] - mcpy[12] * mcpy[3] * mcpy[5];

	inv[14] = -mcpy[0] * mcpy[5] * mcpy[14] + mcpy[0] * mcpy[6] * mcpy[13] + mcpy[4] * mcpy[1] * mcpy[14] -
	          mcpy[4] * mcpy[2] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[6] + mcpy[12] * mcpy[2] * mcpy[5];

	inv[3] = -mcpy[1] * mcpy[6] * mcpy[11] + mcpy[1] * mcpy[7] * mcpy[10] + mcpy[5] * mcpy[2] * mcpy[11] -
	         mcpy[5] * mcpy[3] * mcpy[10] - mcpy[9] * mcpy[2] * mcpy[7] + mcpy[9] * mcpy[3] * mcpy[6];

	inv[7] = mcpy[0] * mcpy[6] * mcpy[11] - mcpy[0] * mcpy[7] * mcpy[10] - mcpy[4] * mcpy[2] * mcpy[11] +
	         mcpy[4] * mcpy[3] * mcpy[10] + mcpy[8] * mcpy[2] * mcpy[7] - mcpy[8] * mcpy[3] * mcpy[6];

	inv[11] = -mcpy[0] * mcpy[5] * mcpy[11] + mcpy[0] * mcpy[7] * mcpy[9] + mcpy[4] * mcpy[1] * mcpy[11] -
	          mcpy[4] * mcpy[3] * mcpy[9] - mcpy[8] * mcpy[1] * mcpy[7] + mcpy[8] * mcpy[3] * mcpy[5];

	inv[15] = mcpy[0] * mcpy[5] * mcpy[10] - mcpy[0] * mcpy[6] * mcpy[9] - mcpy[4] * mcpy[1] * mcpy[10] +
	          mcpy[4] * mcpy[2] * mcpy[9] + mcpy[8] * mcpy[1] * mcpy[6] - mcpy[8] * mcpy[2] * mcpy[5];

	det = mcpy[0] * inv[0] + mcpy[1] * inv[4] + mcpy[2] * inv[8] + mcpy[3] * inv[12];

	if (det == 0) {
		//std::cout << "/n Breaking out the determinant is 0 /n";
	} else {
		det = 1.0 / det;
		if(buffer_select6)
			for (int i = 0; i < 16; i++) bufferinv4[i] = inv[i] * det;
		else
			for (int i = 0; i < 16; i++) _bufferinv4[i] = inv[i] * det;
	}

    }


    {//mmultiply(bufferinv1, bufferinv4, bufferinv5);
        int i, j;
        float Abuf[N][N], Bbuf[N][N];

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        Abuf[i][j] = bufferinv1[i * N + j];
						if(buffer_select6)
							Bbuf[i][j] = bufferinv4[i * N + j];
						else
							Bbuf[i][j] = _bufferinv4[i * N + j];
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        float l_result1 = 0.0;
                        for (int k = 0; k < N; k++) {
                                l_result1 += Abuf[i][k] * Bbuf[k][j];
                        }
						if(buffer_select6)
							bufferinv5[i * N + j] = l_result1;
						else
							_bufferinv5[i * N + j] = l_result1;
                }
        }




    }

    for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
			if(buffer_select6)
				bufferinv5[k * 4 + l] *= -1;
			else
				_bufferinv5[k * 4 + l] *= -1;
        }
    }



    free(inv1);
    //free(inv2);
    //free(intmedt1);
    //free(intmedt2);
    //free(intmedb1);
    //free(intmedb2);
    //free(intmedb3);
    //free(intmedb4);
    //free(buffer1);
    //free(buffer2);
    //free(buffer3);
    //free(buffer4);

    free(bufferinv1);
    free(bufferinv2);
    free(bufferinv3);
	buffer_select6 = !buffer_select6;

}

// mmult4(result1, result1imag, bufferinv4, bufferinv5, result3, result3imag);
extern "C" void TM_mmult4(void) {
        int i, j;
		static bool buffer_select7 = false;
        float Abuf[N][N], Aibuf[N][N], Bbuf[N][N], Bibuf[N][N];
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
					if(buffer_select7){
                        Abuf[i][j] = result1[i * N + j];
                        Aibuf[i][j] = result1imag[i * N + j];
					}
					else{
						Abuf[i][j] = _result1[i * N + j];
                        Aibuf[i][j] = _result1imag[i * N + j];
					}
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
					if(buffer_select7){
                        Bbuf[i][j] = bufferinv4[i * N + j];
                        Bibuf[i][j] = bufferinv5[i * N + j];
					}
					else {
						Bbuf[i][j] = _bufferinv4[i * N + j];
                        Bibuf[i][j] = _bufferinv5[i * N + j];
					}
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        float l_result1 = 0, l_result2 = 0, l_result3 = 0, l_result4 = 0;
                        for (int k = 0; k < N; k++) {
                                float term1 = Abuf[i][k] * Bbuf[k][j];
                                l_result1 += term1;
                                float term2 = Aibuf[i][k] * Bibuf[k][j];
                                l_result2 += term2;
                                float term3 = Abuf[i][k] * Bibuf[k][j];
                                l_result3 += term3;
                                float term4 = Aibuf[i][k] * Bbuf[k][j];
                                l_result4 += term4;
                        }
						if(buffer_select7){
							result3[i * N + j] = l_result1 - l_result2;
							result3imag[i * N + j] = l_result3 + l_result4;
						}
						else{
							_result3[i * N + j] = l_result1 - l_result2;
							_result3imag[i * N + j] = l_result3 + l_result4;
						}
                }
        }
	buffer_select7 = !buffer_select7;
}



extern "C" void TM_mmult64(void) {//mmult64(result3, result3imag, S, Si, result4, result4imag);
        int i, j;
		static bool buffer_select8 = false;
        float Abuf[N][N], Aibuf[N][N], Bbuf[N][M], Bibuf[N][M];
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
					if(buffer_select8){
                        Abuf[i][j] = result3[i * N + j];
                        Aibuf[i][j] = result3imag[i * N + j];
					}
					else {
						Abuf[i][j] = _result3[i * N + j];
                        Aibuf[i][j] = _result3imag[i * N + j];
					}
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
					if(buffer_select8){
                        Bbuf[i][j] = S_inter_buffer[i * M + j];
                        Bibuf[i][j] = Si_inter_buffer[i * M + j];
					}
					else{
						Bbuf[i][j] = _S_inter_buffer[i * M + j];
                        Bibuf[i][j] = _Si_inter_buffer[i * M + j];
					}
                }
        }
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                        float l_result1 = 0, l_result2 = 0, l_result3 = 0, l_result4 = 0;
                        for (int k = 0; k < N; k++) {
                                float term1 = Abuf[i][k] * Bbuf[k][j];
                                l_result1 += term1;
                                float term2 = Aibuf[i][k] * Bibuf[k][j];
                                l_result2 += term2;
                                float term3 = Abuf[i][k] * Bibuf[k][j];
                                l_result3 += term3;
                                float term4 = Aibuf[i][k] * Bbuf[k][j];
                                l_result4 += term4;
                        }
						if(buffer_select8){
							result4[i * M + j] = l_result1 - l_result2;
							result4imag[i * M + j] = l_result3 + l_result4;
						}
						else{
							_result4[i * M + j] = l_result1 - l_result2;
							_result4imag[i * M + j] = l_result3 + l_result4;
						}		
                }
        }
	buffer_select8 = !buffer_select8;
}
// msub(Z, Zi, result4, result4imag, zres, zresimag);
extern "C" void TM_msub(void) {
		static bool buffer_select9 = false;
        float Abuf[N][M], Aibuf[N][M], Bbuf[N][M], Bibuf[N][M];

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
					if(buffer_select9){
                        Abuf[i][j] = Z_inter_buffer[i * M + j];
                        Aibuf[i][j] = Zi_inter_buffer[i * M + j];
                        Bbuf[i][j] = result4[i * M + j];
                        Bibuf[i][j] = result4imag[i * M + j];
					}
					else{
						Abuf[i][j] = _Z_inter_buffer[i * M + j];
                        Aibuf[i][j] = _Zi_inter_buffer[i * M + j];
                        Bbuf[i][j] = _result4[i * M + j];
                        Bibuf[i][j] = _result4imag[i * M + j];
					}
                }
        }

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
					if(buffer_select9){
                        zres[i * M + j] = float(Abuf[i][j] - Bbuf[i][j]);
                        zresimag[i * M + j] = float(Aibuf[i][j] - Bibuf[i][j]);
					}
					else{
						_zres[i * M + j] = float(Abuf[i][j] - Bbuf[i][j]);
                        _zresimag[i * M + j] = float(Aibuf[i][j] - Bibuf[i][j]);
					}
                }
        }
	buffer_select9 = !buffer_select9;
}


extern "C" void TM_display_result(void) {
	static bool buffer_select10 = false;
    if (frame_it < 5){
    	std::ofstream outfile;
    	outfile.open("cedr_TM_output.txt", std::ios_base::app);
    	outfile <<"Frame id:" << frame_it<< " Real part: \n";
		//std::cout << "*****Final result being printed for frame number "<<frame_it <<" ******\n";
		//std::cout <<"********buffer select is " << buffer_select10 << "*****************\n";
		//std::cout << "Real part: \n";
		for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 64; j++) {
						
						if(buffer_select10){
							outfile<< zres[i * 64 + j] << " ";
							//std::cout << zres[i * 64 + j] << " ";
						}
						else{
							outfile<< _zres[i * 64 + j] << " ";
							//std::cout << _zres[i * 64 + j] << " ";
						}
				}
				//std::cout << "\n";
				outfile<< "\n";
		}
		//std::cout << "Imag part: Here\n";
		outfile << "Frame id:" << frame_it<< " Imag part: \n";
		for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 64; j++) {
				
					if(buffer_select10){
						outfile<< zresimag[i * 64 + j] << " ";
						//std::cout << zresimag[i * 64 + j] << " ";
					}
					else{
						outfile<< _zresimag[i * 64 + j] << " ";
						//std::cout << _zresimag[i * 64 + j] << " ";
					}
				}
				//std::cout << "\n";
				outfile << "\n";
		}
		//fflush(stdout);
		//usleep(10000);
    	outfile.close();
    }
    frame_it++;
	buffer_select10 = !buffer_select10;
}

int main(void) {}
