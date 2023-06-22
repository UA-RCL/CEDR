#include"inverse.h"


void msub(float A[N*M],float Ai[N*M], float B[N*M], float Bi[N*M] ,float C[N*M], float Ci[N*M])
{
	float Abuf[N][M],Aibuf[N][M],Bbuf[N][M],Bibuf[N][M],Cbuf[N][M],Cibuf[N][M];

	 for(int i=0; i<N; i++) {
	          for(int j=0; j<M; j++) {
// 	#pragma HLS PIPELINE
	               Abuf[i][j] = A[i * M + j];
	               Aibuf[i][j] = Ai[i*M + j];
	               Bbuf[i][j] = B[i*M + j];
	               Bibuf[i][j] = Bi[i*M + j];
	               Bbuf[i][j]*=-1;
	               Bibuf[i][j]*=-1;

	          }
	     }

	 for(int i=0; i<N; i++) {
	 	          for(int j=0; j<M; j++) {
// #pragma HLS RESOURCE variable=Cbuf core=AddSub_DSP
// #pragma HLS RESOURCE variable=Cibuf core=AddSub_DSP
// 	 	        #pragma HLS PIPELINE
	 	        	  Cbuf[i][j]=Abuf[i][j] + Bbuf[i][j];
	 	        	  Cibuf[i][j]=Aibuf[i][j] + Bibuf[i][j];
	 	        	  C[i * M + j]=Cbuf[i][j];
	 	        	  Ci[i * M + j]=Cibuf[i][j];

	 	      }
	     }





}
