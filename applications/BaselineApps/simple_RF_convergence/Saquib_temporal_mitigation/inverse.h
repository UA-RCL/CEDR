//#ifndef _INVERSE_H_
//#define _INVERSE_H_


//#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
//#include<iostream>
#include<math.h>
//#include<complex>


// #include<bits/stdc++.h>
// using namespace std;
//#ifndef _INVERSE_H_
//#define _INVERSE_H_

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

//void inverse()


//#endif


