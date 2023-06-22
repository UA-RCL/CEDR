#include "inverse.h"

void mmultiply(float A[N*N],float B[N*N],float C[N*N])
{
	int i,j;
	float Abuf[N][N],Bbuf[N][N];
// #pragma HLS array_partition variable=Abuf block factor=4 dim=2
// #pragma HLS array_partition variable=Bbuf block factor=4 dim=1

	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
// #pragma HLS PIPELINE
			Abuf[i][j]=A[i*N+j];
			Bbuf[i][j]=B[i*N+j];
		}
	}

	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
// #pragma HLS PIPELINE
			float result1=0.0;
			for(int k=0;k<N;k++){
				result1+=Abuf[i][k]*Bbuf[k][j];
			}
			C[i*N+j]=result1;
		}
	}


}
