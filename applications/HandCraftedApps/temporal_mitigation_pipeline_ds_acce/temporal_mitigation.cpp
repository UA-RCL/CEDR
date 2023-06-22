
#include "temporal_mitigation.hpp"
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#define PROGPATH "./input/"
#define ZIN PROGPATH "z.txt"
#define ZIMAGIN PROGPATH "zimag.txt"
#define SIN PROGPATH "s.txt"
#define SIMAGIN PROGPATH "simag.txt"
#include <dlfcn.h>
#include <stdio.h>
#include <fstream>

int TM_iter;
double *Z, *Zi;
double *S, *Si;
double *Z_inter_buffer, *Zi_inter_buffer;
double *S_inter_buffer, *Si_inter_buffer;
double *Shermitian, *Shermitianimag;
int M, N;
double *result1, *result1imag;
double *result2, *result2imag;
double *bufferinv4, *bufferinv5;
double *result3, *result3imag;
double *result4, *result4imag;
double *zres, *zresimag;
bool buffer_select4;
bool buffer_select5;
int frame_it;

double *_Z, *_Zi;
double *_S, *_Si;
double *_Z_inter_buffer, *_Zi_inter_buffer;
double *_S_inter_buffer, *_Si_inter_buffer;
double *_Shermitian, *_Shermitianimag;
double *_result1, *_result1imag;
double *_result2, *_result2imag;
double *_bufferinv4, *_bufferinv5;
double *_result3, *_result3imag;
double *_result4, *_result4imag;
double *_zres, *_zresimag;

// Pointer to use to hold the shared object file handle
void *dlhandle;
void (*mmult_accel_func)(double**, double**, double**, double**, double**, double**, size_t*, size_t*, size_t*, uint8_t);

__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_CLUSTER_IDX__ = 0;

void __attribute__((constructor)) setup(void) {
  printf("[Temporal Mitigation] initializing buffers\n");

  TM_iter = 0;
  N = 4;
  M = 64;
  frame_it = 0;
  buffer_select4 = false;
  buffer_select5 = false;
  // FILE *Zreal, *Zimag, *Sreal, *Simag;

  // Initializing the Z signal which will have 4*64 dimension
  Z = (double *)malloc(N * M * sizeof(double));
  Zi = (double *)malloc(N * M * sizeof(double));
  _Z = (double *)malloc(N * M * sizeof(double));
  _Zi = (double *)malloc(N * M * sizeof(double));
  Z_inter_buffer = (double *)malloc(N * M * sizeof(double));
  Zi_inter_buffer = (double *)malloc(N * M * sizeof(double));
  _Z_inter_buffer = (double *)malloc(N * M * sizeof(double));
  _Zi_inter_buffer = (double *)malloc(N * M * sizeof(double));
  // Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted
  // by S
  S = (double *)malloc(N * M * sizeof(double));
  Si = (double *)malloc(N * M * sizeof(double));
  _S = (double *)malloc(N * M * sizeof(double));
  _Si = (double *)malloc(N * M * sizeof(double));
  S_inter_buffer = (double *)malloc(N * M * sizeof(double));
  Si_inter_buffer = (double *)malloc(N * M * sizeof(double));
  _S_inter_buffer = (double *)malloc(N * M * sizeof(double));
  _Si_inter_buffer = (double *)malloc(N * M * sizeof(double));
  // now defining the argument files which will contain the corresponding values of Z and S
  // Zreal = fopen(ZIN, "r");
  // Zimag = fopen(ZIMAGIN, "r");
  // Sreal = fopen(SIN, "r");
  // Simag = fopen(SIMAGIN, "r");

  //// now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
  // for (int i = 0; i < N; i++) {
  //    for (int j = 0; j < M; j++) {
  //        fscanf(Zreal, "%f", &Z[i * M + j]); Z[i * M + j] /= 10.0f;
  //        fscanf(Zimag, "%f", &Zi[i * M + j]); Zi[i * M + j] /= 10.0f;
  //        fscanf(Sreal, "%f", &S[i * M + j]); S[i * M + j] /= 10.0f;
  //        fscanf(Simag, "%f", &Si[i * M + j]); Si[i * M + j] /= 10.0f;
  //    }
  //}

  //// Computing the hermitian of S
  Shermitian = (double *)malloc(M * N * sizeof(double));
  Shermitianimag = (double *)malloc(M * N * sizeof(double));
  result1 = (double *)malloc(N * N * sizeof(double));
  result1imag = (double *)malloc(N * N * sizeof(double));

  result2 = (double *)malloc(N * N * sizeof(double));
  result2imag = (double *)malloc(N * N * sizeof(double));

  bufferinv4 = (double *)malloc(N * N * sizeof(double));
  bufferinv5 = (double *)malloc(N * N * sizeof(double));

  result3 = (double *)malloc(N * N * sizeof(double));
  result3imag = (double *)malloc(N * N * sizeof(double));

  result4 = (double *)malloc(N * M * sizeof(double));
  result4imag = (double *)malloc(N * M * sizeof(double));

  zres = (double *)malloc(N * M * sizeof(double));
  zresimag = (double *)malloc(N * M * sizeof(double));

  _Shermitian = (double *)malloc(M * N * sizeof(double));
  _Shermitianimag = (double *)malloc(M * N * sizeof(double));
  _result1 = (double *)malloc(N * N * sizeof(double));
  _result1imag = (double *)malloc(N * N * sizeof(double));

  _result2 = (double *)malloc(N * N * sizeof(double));
  _result2imag = (double *)malloc(N * N * sizeof(double));

  _bufferinv4 = (double *)malloc(N * N * sizeof(double));
  _bufferinv5 = (double *)malloc(N * N * sizeof(double));

  _result3 = (double *)malloc(N * N * sizeof(double));
  _result3imag = (double *)malloc(N * N * sizeof(double));

  _result4 = (double *)malloc(N * M * sizeof(double));
  _result4imag = (double *)malloc(N * M * sizeof(double));

  _zres = (double *)malloc(N * M * sizeof(double));
  _zresimag = (double *)malloc(N * M * sizeof(double));

  ///////////////accelerator////////////////////////////////
  dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open libdash-rt shared object!\n");
  } else {
    mmult_accel_func =
      (void (*)(double**, double**, double**, double**, double**, double**, size_t*, size_t*, size_t*, uint8_t)) dlsym(dlhandle, "DASH_GEMM_gemm");
    if (mmult_accel_func == nullptr) {
      printf("Unable to get function handle for DASH_GEMM_gemm accelerator function!\n");
    }
  }

  remove("cedr_TM_output.txt");
  printf("[Temporal Mitigation] initialization complete\n");
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

  dlclose(dlhandle);
  printf("[Temporal mitigation] buffers destroyed\n");
}

extern "C" void TM_head_node(void) {
  static bool buffer_select0 = false;
  FILE *Zreal, *Zimag, *Sreal, *Simag;
  // FILE *_Zreal, *_Zimag, *_Sreal, *_Simag;
  Zreal = fopen(ZIN, "r");
  Zimag = fopen(ZIMAGIN, "r");
  Sreal = fopen(SIN, "r");
  Simag = fopen(SIMAGIN, "r");

  printf("[Temporal Mitigation] Reading one frame's worth of input data\n");

  // _Zreal = fopen(ZIN, "r");
  // _Zimag = fopen(ZIMAGIN, "r");
  // _Sreal = fopen(SIN, "r");
  // _Simag = fopen(SIMAGIN, "r");

  // now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (buffer_select0) {
        fscanf(Zreal, "%lf", &Z[i * M + j]);
        fscanf(Zimag, "%lf", &Zi[i * M + j]);
        fscanf(Sreal, "%lf", &S[i * M + j]);
        fscanf(Simag, "%lf", &Si[i * M + j]);
        Z[i * M + j] = (Z[i * M + j]) / 10.0f;
        Zi[i * M + j] = (Zi[i * M + j]) / 10.0f;
        S[i * M + j] = (S[i * M + j]) / 10.0f;
        Si[i * M + j] = (Si[i * M + j]) / 10.0f;
      } else {
        // fscanf(_Zreal, "%f", &_Z[i * M + j]);
        // fscanf(_Zimag, "%f", &_Zi[i * M + j]);
        // fscanf(_Sreal, "%f", &_S[i * M + j]);
        // fscanf(_Simag, "%f", &_Si[i * M + j]);
        fscanf(Zreal, "%lf", &_Z[i * M + j]);
        fscanf(Zimag, "%lf", &_Zi[i * M + j]);
        fscanf(Sreal, "%lf", &_S[i * M + j]);
        fscanf(Simag, "%lf", &_Si[i * M + j]);
        _Z[i * M + j] = (_Z[i * M + j]) / 10.0f;
        _Zi[i * M + j] = (_Zi[i * M + j]) / 10.0f;
        _S[i * M + j] = (_S[i * M + j]) / 10.0f;
        _Si[i * M + j] = (_Si[i * M + j]) / 10.0f;
      }
    }
  }
  // if(buffer_select0){
  // printf("%lf , %lf ", Z[0], Zi[0]);
  // }
  // else {
  // printf("%lf , %lf ", _Z[0], _Zi[0]);
  // }
  fclose(Zreal);
  fclose(Zimag);
  fclose(Sreal);
  fclose(Simag);
  // fclose(_Zreal);
  // fclose(_Zimag);
  // fclose(_Sreal);
  // fclose(_Simag);
  // TM_iter++;
  buffer_select0 = !buffer_select0;
}

extern "C" void TM_hermitian(void) {
  static bool buffer_select1 = false;

  printf("[Temporal Mitigation] Executing hermitian transpose\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (buffer_select1) {
        Shermitian[j * N + i] = S[i * M + j];
        Shermitianimag[j * N + i] = -Si[i * M + j];
      } else {
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

extern "C" void TM_Z_buffer(void) {
  static bool buffer_select2 = false;
  for (int i = 0; i < N * M; i++) {
    if (buffer_select2) {
      Z_inter_buffer[i] = Z[i];
      Zi_inter_buffer[i] = Zi[i];
    } else {
      _Z_inter_buffer[i] = _Z[i];
      _Zi_inter_buffer[i] = _Zi[i];
    }
  }
  buffer_select2 = !buffer_select2;
}
extern "C" void TM_S_buffer(void) {
  static bool buffer_select3 = false;
  for (int i = 0; i < N * M; i++) {
    if (buffer_select3) {
      S_inter_buffer[i] = S[i];
      Si_inter_buffer[i] = Si[i];
    } else {
      _S_inter_buffer[i] = _S[i];
      _Si_inter_buffer[i] = _Si[i];
    }
  }
  buffer_select3 = !buffer_select3;
}

extern "C" void TM_mmult_Z_cpu(void) {
  double res1 = 0, res2 = 0, res3 = 0, res4 = 0;
  double term1, term2, term3, term4;

  printf("[Temporal Mitigation] Executing MMULT on CPU\n");

  // printf("\n######################Z Matrix Before the MMULT CPU######################\n");
  // if(buffer_select4){
  // printf("%lf , %lf \n", Z[0], Zi[0]);
  // }
  // else{
  // printf("%lf , %lf \n", _Z[0], _Zi[0]);
  // }
  // printf("##################################################################\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      for (int k = 0; k < M; k++) {
        if (buffer_select4) {
          // double term1 = Abuf[i][k] * Bbuf[k][j];
          term1 = Z[i * M + k] * Shermitian[k * N + j];
          res1 += term1;
          // double term2 = Aibuf[i][k] * Bibuf[k][j];
          term2 = Zi[i * M + k] * Shermitianimag[k * N + j];
          res2 += term2;
          // double term3 = Abuf[i][k] * Bibuf[k][j];
          term3 = Z[i * M + k] * Shermitianimag[k * N + j];
          res3 += term3;
          // double term4 = Aibuf[i][k] * Bbuf[k][j];
          term4 = Zi[i * M + k] * Shermitian[k * N + j];
          res4 += term4;
        } else {
          // double term1 = Abuf[i][k] * Bbuf[k][j];
          term1 = _Z[i * M + k] * _Shermitian[k * N + j];
          res1 += term1;
          // double term2 = Aibuf[i][k] * Bibuf[k][j];
          term2 = _Zi[i * M + k] * _Shermitianimag[k * N + j];
          res2 += term2;
          // double term3 = Abuf[i][k] * Bibuf[k][j];
          term3 = _Z[i * M + k] * _Shermitianimag[k * N + j];
          res3 += term3;
          // double term4 = Aibuf[i][k] * Bbuf[k][j];
          term4 = _Zi[i * M + k] * _Shermitian[k * N + j];
          res4 += term4;
        }
      }
      if (buffer_select4) {
        result1[i * N + j] = res1 - res2;
        result1imag[i * N + j] = res3 + res4;
      } else {
        _result1[i * N + j] = res1 - res2;
        _result1imag[i * N + j] = res3 + res4;
      }
    }
  }

  /*
  printf("\n###################### Z * Shermitian (CPU) ######################\n");
  if (buffer_select4) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", result1[i * N + j], result1imag[i * N + j]);
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", _result1[i * N + j], _result1imag[i * N + j]);
      }
    }
  }
  printf("##################################################################\n");
  */
  buffer_select4 = !buffer_select4;
}

extern "C" void TM_mmult_Z_acce(void) {
  /*
  printf("\n######################Z Matrix Before the MMULT ACCE######################\n");
  if (buffer_select4) {
    printf("%lf , %lf \n", Z[0], Zi[0]);
  } else {
    printf("%lf , %lf \n", _Z[0], _Zi[0]);
  }
  printf("##################################################################\n");
  */

  printf("[Temporal Mitigation] Executing MMULT on Accelerator\n");
  size_t A_ROW = 4;
  size_t A_COL = 64;
  size_t B_COL = 4;

  if (buffer_select4) {
    (*mmult_accel_func)(&Z, &Shermitian, &Zi, &Shermitianimag, &result1, &result1imag, &A_ROW, &A_COL, &B_COL, __CEDR_CLUSTER_IDX__);
  } else {
    (*mmult_accel_func)(&_Z, &_Shermitian, &_Zi, &_Shermitianimag, &_result1, &_result1imag, &A_ROW, &A_COL, &B_COL, __CEDR_CLUSTER_IDX__);
  }

  /*
  printf("\n###################### Z * Shermitian (Accelerator) ######################\n");
  if (buffer_select4) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", result1[i * N + j], result1imag[i * N + j]);
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", _result1[i * N + j], _result1imag[i * N + j]);
      }
    }
  }
  printf("##################################################################\n");
  */
  buffer_select4 = !buffer_select4;
}

extern "C" void TM_mmult_S_acce(void) {

  printf("[Temporal Mitigation] Executing MMULT on Accelerator\n");

  size_t A_ROW = 4;
  size_t A_COL = 64;
  size_t B_COL = 4;

  if (buffer_select5) {
    (*mmult_accel_func)(&S, &Shermitian, &Si, &Shermitianimag, &result2, &result2imag, &A_ROW, &A_COL, &B_COL, __CEDR_CLUSTER_IDX__);
  } else {
    (*mmult_accel_func)(&_S, &_Shermitian, &_Si, &_Shermitianimag, &_result2, &_result2imag, &A_ROW, &A_COL, &B_COL, __CEDR_CLUSTER_IDX__);
  }

  //  printf("\n###################### S * Shermitian (Accelerator) ######################\n");
  //  if (buffer_select5) {
  //    for (int i = 0; i < N; i++) {
  //      for (int j = 0; j < N; j++) {
  //        printf("%lf , %lf \n", result2[i * N + j], result2imag[i * N + j]);
  //      }
  //    }
  //  } else {
  //    for (int i = 0; i < N; i++) {
  //      for (int j = 0; j < N; j++) {
  //        printf("%lf , %lf \n", _result2[i * N + j], _result2imag[i * N + j]);
  //      }
  //    }
  //  }
  //  printf("##################################################################\n");
  buffer_select5 = !buffer_select5;
}

extern "C" void TM_mmult_S_cpu(void) {
  double res1 = 0, res2 = 0, res3 = 0, res4 = 0;
  double term1, term2, term3, term4;

  printf("[Temporal Mitigation] Executing MMULT on CPU\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      for (int k = 0; k < M; k++) {
        if (buffer_select5) {
          // printf("%f  ", S[i * M + k] );
          // double term1 = Abuf[i][k] * Bbuf[k][j];
          term1 = S[i * M + k] * Shermitian[k * N + j];
          res1 += term1;
          // double term2 = Aibuf[i][k] * Bibuf[k][j];
          term2 = Si[i * M + k] * Shermitianimag[k * N + j];
          res2 += term2;
          // double term3 = Abuf[i][k] * Bibuf[k][j];
          term3 = S[i * M + k] * Shermitianimag[k * N + j];
          res3 += term3;
          // double term4 = Aibuf[i][k] * Bbuf[k][j];
          term4 = Si[i * M + k] * Shermitian[k * N + j];
          res4 += term4;
        } else {
          // double term1 = Abuf[i][k] * Bbuf[k][j];
          term1 = _S[i * M + k] * _Shermitian[k * N + j];
          res1 += term1;
          // double term2 = Aibuf[i][k] * Bibuf[k][j];
          term2 = _Si[i * M + k] * _Shermitianimag[k * N + j];
          res2 += term2;
          // double term3 = Abuf[i][k] * Bibuf[k][j];
          term3 = _S[i * M + k] * _Shermitianimag[k * N + j];
          res3 += term3;
          // double term4 = Aibuf[i][k] * Bbuf[k][j];
          term4 = _Si[i * M + k] * _Shermitian[k * N + j];
          res4 += term4;
        }
      }
      if (buffer_select5) {
        result2[i * N + j] = res1 - res2;
        result2imag[i * N + j] = res3 + res4;
      } else {
        _result2[i * N + j] = res1 - res2;
        _result2imag[i * N + j] = res3 + res4;
      }
    }
  }
  /*
  printf("\n###################### S * Shermitian (CPU) ######################\n");
  if (buffer_select5) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", result2[i * N + j], result2imag[i * N + j]);
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        printf("%lf , %lf \n", _result2[i * N + j], _result2imag[i * N + j]);
      }
    }
  }
  printf("##################################################################\n");
   */
  buffer_select5 = !buffer_select5;
}

extern "C" void TM_inverse(void) {
  static bool buffer_select6 = false;

  printf("[Temporal Mitigation] Computing matrix inverse\n");

  double *inv1;
  double *bufferinv1, *bufferinv2, *bufferinv3;
  inv1 = (double *)malloc(N * N * sizeof(double));
  // inv2 = (double *)malloc(N * N * sizeof(double));
  // intmedt1 = (double *)malloc(N * N * sizeof(double));
  // intmedt2 = (double *)malloc(N * N * sizeof(double));
  // intmedb1 = (double *)malloc(N * N * sizeof(double));
  // intmedb2 = (double *)malloc(N * N * sizeof(double));
  // intmedb3 = (double *)malloc(N * N * sizeof(double));
  // intmedb4 = (double *)malloc(N * N * sizeof(double));
  // buffer1 = (double *)malloc(N * N * sizeof(double));
  // buffer2 = (double *)malloc(N * N * sizeof(double));
  // buffer3 = (double *)malloc(N * N * sizeof(double));
  // buffer4 = (double *)malloc(N * N * sizeof(double));
  // The following arrays are used for the inverse computation

  bufferinv1 = (double *)malloc(N * N * sizeof(double));
  bufferinv2 = (double *)malloc(N * N * sizeof(double));
  bufferinv3 = (double *)malloc(N * N * sizeof(double));
  { // alternateinverse(result2, inv1);
    double inv[16], det;
    int i;

    double mcpy[16];
    for (int i = 0; i < 16; i++) {
      if (buffer_select6)
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
      // std::cout << "/n Breaking out the determinant is 0 /n";
    } else {
      det = 1.0 / det;

      for (int i = 0; i < 16; i++)
        inv1[i] = inv[i] * det;
    }
  }

  { // mmultiply(inv1, result2imag, bufferinv1);
    int i, j;
    double Abuf[N][N], Bbuf[N][N];

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        Abuf[i][j] = inv1[i * N + j];
        if (buffer_select6)
          Bbuf[i][j] = result2imag[i * N + j];
        else
          Bbuf[i][j] = _result2imag[i * N + j];
      }
    }

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double result1 = 0.0;
        for (int k = 0; k < N; k++) {
          result1 += Abuf[i][k] * Bbuf[k][j];
        }
        bufferinv1[i * N + j] = result1;
      }
    }
  }

  { // mmultiply(result2imag, bufferinv1, bufferinv2);

    int i, j;
    double Abuf[N][N], Bbuf[N][N];

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (buffer_select6)
          Abuf[i][j] = result2imag[i * N + j];
        else
          Abuf[i][j] = _result2imag[i * N + j];
        Bbuf[i][j] = bufferinv1[i * N + j];
      }
    }

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double l_result1 = 0.0;
        for (int k = 0; k < N; k++) {
          l_result1 += Abuf[i][k] * Bbuf[k][j];
        }
        bufferinv2[i * N + j] = l_result1;
      }
    }
  }

  { // mmadd(bufferinv2, result2, bufferinv3);
    double Abuf[N][N], Bbuf[N][N];

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        Abuf[i][j] = bufferinv2[i * N + j];
        //                     Aibuf[i][j] = Ai[i*M + j];
        if (buffer_select6)
          Bbuf[i][j] = result2[i * N + j];
        else
          Bbuf[i][j] = _result2[i * N + j];
        //                     Bibuf[i][j] = Bi[i*M + j];
      }
    }

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        bufferinv3[i * N + j] = double(Abuf[i][j] + Bbuf[i][j]);
      }
    }
  }

  { // alternateinverse(bufferinv3, bufferinv4);

    double inv[16], det;
    int i;

    double mcpy[16];
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
      // std::cout << "/n Breaking out the determinant is 0 /n";
    } else {
      det = 1.0 / det;
      if (buffer_select6)
        for (int i = 0; i < 16; i++)
          bufferinv4[i] = inv[i] * det;
      else
        for (int i = 0; i < 16; i++)
          _bufferinv4[i] = inv[i] * det;
    }
  }

  { // mmultiply(bufferinv1, bufferinv4, bufferinv5);
    int i, j;
    double Abuf[N][N], Bbuf[N][N];

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        Abuf[i][j] = bufferinv1[i * N + j];
        if (buffer_select6)
          Bbuf[i][j] = bufferinv4[i * N + j];
        else
          Bbuf[i][j] = _bufferinv4[i * N + j];
      }
    }

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double l_result1 = 0.0;
        for (int k = 0; k < N; k++) {
          l_result1 += Abuf[i][k] * Bbuf[k][j];
        }
        if (buffer_select6)
          bufferinv5[i * N + j] = l_result1;
        else
          _bufferinv5[i * N + j] = l_result1;
      }
    }
  }

  for (int k = 0; k < 4; k++) {
    for (int l = 0; l < 4; l++) {
      if (buffer_select6)
        bufferinv5[k * 4 + l] *= -1;
      else
        _bufferinv5[k * 4 + l] *= -1;
    }
  }

  free(inv1);
  // free(inv2);
  // free(intmedt1);
  // free(intmedt2);
  // free(intmedb1);
  // free(intmedb2);
  // free(intmedb3);
  // free(intmedb4);
  // free(buffer1);
  // free(buffer2);
  // free(buffer3);
  // free(buffer4);

  free(bufferinv1);
  free(bufferinv2);
  free(bufferinv3);
  buffer_select6 = !buffer_select6;
}

// mmult4(result1, result1imag, bufferinv4, bufferinv5, result3, result3imag);
extern "C" void TM_mmult4(void) {
  int i, j;
  static bool buffer_select7 = false;
  double Abuf[N][N], Aibuf[N][N], Bbuf[N][N], Bibuf[N][N];

  printf("[Temporal Mitigation] Executing MMULT on CPU\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (buffer_select7) {
        Abuf[i][j] = result1[i * N + j];
        Aibuf[i][j] = result1imag[i * N + j];
      } else {
        Abuf[i][j] = _result1[i * N + j];
        Aibuf[i][j] = _result1imag[i * N + j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (buffer_select7) {
        Bbuf[i][j] = bufferinv4[i * N + j];
        Bibuf[i][j] = bufferinv5[i * N + j];
      } else {
        Bbuf[i][j] = _bufferinv4[i * N + j];
        Bibuf[i][j] = _bufferinv5[i * N + j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double l_result1 = 0, l_result2 = 0, l_result3 = 0, l_result4 = 0;
      for (int k = 0; k < N; k++) {
        double term1 = Abuf[i][k] * Bbuf[k][j];
        l_result1 += term1;
        double term2 = Aibuf[i][k] * Bibuf[k][j];
        l_result2 += term2;
        double term3 = Abuf[i][k] * Bibuf[k][j];
        l_result3 += term3;
        double term4 = Aibuf[i][k] * Bbuf[k][j];
        l_result4 += term4;
      }
      if (buffer_select7) {
        result3[i * N + j] = l_result1 - l_result2;
        result3imag[i * N + j] = l_result3 + l_result4;
      } else {
        _result3[i * N + j] = l_result1 - l_result2;
        _result3imag[i * N + j] = l_result3 + l_result4;
      }
    }
  }
  buffer_select7 = !buffer_select7;
}

extern "C" void TM_mmult64(void) { // mmult64(result3, result3imag, S, Si, result4, result4imag);
  int i, j;
  static bool buffer_select8 = false;
  double Abuf[N][N], Aibuf[N][N], Bbuf[N][M], Bibuf[N][M];

  printf("[Temporal Mitigation] Executing MMULT on CPU\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (buffer_select8) {
        Abuf[i][j] = result3[i * N + j];
        Aibuf[i][j] = result3imag[i * N + j];
      } else {
        Abuf[i][j] = _result3[i * N + j];
        Aibuf[i][j] = _result3imag[i * N + j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (buffer_select8) {
        Bbuf[i][j] = S_inter_buffer[i * M + j];
        Bibuf[i][j] = Si_inter_buffer[i * M + j];
      } else {
        Bbuf[i][j] = _S_inter_buffer[i * M + j];
        Bibuf[i][j] = _Si_inter_buffer[i * M + j];
      }
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      double l_result1 = 0, l_result2 = 0, l_result3 = 0, l_result4 = 0;
      for (int k = 0; k < N; k++) {
        double term1 = Abuf[i][k] * Bbuf[k][j];
        l_result1 += term1;
        double term2 = Aibuf[i][k] * Bibuf[k][j];
        l_result2 += term2;
        double term3 = Abuf[i][k] * Bibuf[k][j];
        l_result3 += term3;
        double term4 = Aibuf[i][k] * Bbuf[k][j];
        l_result4 += term4;
      }
      if (buffer_select8) {
        result4[i * M + j] = l_result1 - l_result2;
        result4imag[i * M + j] = l_result3 + l_result4;
      } else {
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
  double Abuf[N][M], Aibuf[N][M], Bbuf[N][M], Bibuf[N][M];

  printf("[Temporal Mitigation] Executing Matrix Subtraction\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (buffer_select9) {
        Abuf[i][j] = Z_inter_buffer[i * M + j];
        Aibuf[i][j] = Zi_inter_buffer[i * M + j];
        Bbuf[i][j] = result4[i * M + j];
        Bibuf[i][j] = result4imag[i * M + j];
      } else {
        Abuf[i][j] = _Z_inter_buffer[i * M + j];
        Aibuf[i][j] = _Zi_inter_buffer[i * M + j];
        Bbuf[i][j] = _result4[i * M + j];
        Bibuf[i][j] = _result4imag[i * M + j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (buffer_select9) {
        zres[i * M + j] = double(Abuf[i][j] - Bbuf[i][j]);
        zresimag[i * M + j] = double(Aibuf[i][j] - Bibuf[i][j]);
      } else {
        _zres[i * M + j] = double(Abuf[i][j] - Bbuf[i][j]);
        _zresimag[i * M + j] = double(Aibuf[i][j] - Bibuf[i][j]);
      }
    }
  }
  buffer_select9 = !buffer_select9;
}

extern "C" void TM_display_result(void) {
  static bool buffer_select10 = false;

  printf("[Temporal Mitigation] Dumping output to file\n");

  //if (frame_it < 5) {
    std::ofstream outfile;
    outfile.open("cedr_TM_output.txt", std::ios_base::app);
    outfile << "Frame id:" << frame_it << " Real part: \n";
    // std::cout << "*****Final result being printed for frame number "<<frame_it <<" ******\n";
    // std::cout <<"********buffer select is " << buffer_select10 << "*****************\n";
    // std::cout << "Real part: \n";
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 64; j++) {

        if (buffer_select10) {
          outfile << zres[i * 64 + j] << " ";
          // std::cout << zres[i * 64 + j] << " ";
        } else {
          outfile << _zres[i * 64 + j] << " ";
          // std::cout << _zres[i * 64 + j] << " ";
        }
      }
      // std::cout << "\n";
      outfile << "\n";
    }
    // std::cout << "Imag part: Here\n";
    outfile << "Frame id:" << frame_it << " Imag part: \n";
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 64; j++) {

        if (buffer_select10) {
          outfile << zresimag[i * 64 + j] << " ";
          // std::cout << zresimag[i * 64 + j] << " ";
        } else {
          outfile << _zresimag[i * 64 + j] << " ";
          // std::cout << _zresimag[i * 64 + j] << " ";
        }
      }
      // std::cout << "\n";
      outfile << "\n";
    }
    // fflush(stdout);
    // usleep(10000);
    outfile.close();
  //}
  printf("[Temporal Mitigation] Temporal mitigation complete!\n");
  frame_it++;
  buffer_select10 = !buffer_select10;
}