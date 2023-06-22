#ifndef _INVERSE_H_
#define _INVERSE_H_

#include <bits/stdc++.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <complex>
#include <iostream>

using namespace std;
//#ifndef _INVERSE_H_
//#define _INVERSE_H_

#define N 4
#define M 64

void getCofactor(double A[N * N], double temp[N * N], int p, int q, int n);
#pragma SDS data data_mover(m : AXIDMA_SIMPLE, invOut : AXIDMA_SIMPLE)
void alternateinverse(double m[16], double invOut[16]);
//#pragma SDS data data_mover(m:AXIDMA_SIMPLE,invOut:AXIDMA_SIMPLE)
void scalableinverse(double m[16], double invOut[16]);
double determinant(double A[N * N], int n);
void adjoint(double A[N * N], double adj[N][N]);
void inverse(double A[N * N], double inverse[N * N]);
void display(double *A, int, int);
// void realpart(double A[N*N], double B[N*N] , double inv1[N*N] ,double inv2[N*N] , double intmedt1[N*N]);
// void imagpart(double A[N*N],double B[N*N],double inv1[N*N], double inv2[N*N],double intmedt2[N*N]);
#pragma SDS data access_pattern(A : SEQUENTIAL, B : SEQUENTIAL, C : SEQUENTIAL)
void mmultiply(double A[N * N], double B[N * N], double C[N * N]);
//#pragma SDS data access_pattern(S:SEQUENTIAL ,Si:SEQUENTIAL,Shermitian:SEQUENTIAL , Shermitianimag:SEQUENTIAL)
void hermitian(double S[N * M], double Si[N * M], double Shermitian[N * M], double Shermitianimag[N * M]);
double divide(double, double);

//#pragma SDS data access_pattern(A:SEQUENTIAL,B:SEQUENTIAL,C:SEQUENTIAL)
void mmadd(double A[N * N], double B[N * N], double C[N * N]);

#pragma SDS data access_pattern(A                \
                                : SEQUENTIAL, Ai \
                                : SEQUENTIAL, B  \
                                : SEQUENTIAL, Bi \
                                : SEQUENTIAL, C  \
                                : SEQUENTIAL, Ci \
                                : SEQUENTIAL)
void mmult(double A[N * N], double Ai[N * N], double B[N * N], double Bi[N * N], double C[N * N], double Ci[N * N]);

#pragma SDS data access_pattern(A                \
                                : SEQUENTIAL, Ai \
                                : SEQUENTIAL, B  \
                                : SEQUENTIAL, Bi \
                                : SEQUENTIAL, C  \
                                : SEQUENTIAL, Ci \
                                : SEQUENTIAL)
void mmult4(double A[N * N], double Ai[N * N], double B[N * N], double Bi[N * N], double C[N * N], double Ci[N * N]);

#pragma SDS data access_pattern(A                \
                                : SEQUENTIAL, Ai \
                                : SEQUENTIAL, B  \
                                : SEQUENTIAL, Bi \
                                : SEQUENTIAL, C  \
                                : SEQUENTIAL, Ci \
                                : SEQUENTIAL)
void mmult64(double A[N * N], double Ai[N * N], double B[N * M], double Bi[N * M], double C[N * M], double Ci[N * M]);

#pragma SDS data access_pattern(A                \
                                : SEQUENTIAL, Ai \
                                : SEQUENTIAL, B  \
                                : SEQUENTIAL, Bi \
                                : SEQUENTIAL, C  \
                                : SEQUENTIAL, Ci \
                                : SEQUENTIAL)
void msub(double A[N * M], double Ai[N * M], double B[N * M], double Bi[N * M], double C[N * M], double Ci[N * M]);

// void inverse()

#endif
