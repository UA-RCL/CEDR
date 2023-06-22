#include"inverse.h"


void mmadd(float A[N*N],float B[N*N],float C[N*N])
{
	float Abuf[N][N],Bbuf[N][N];

	 for(int i=0; i<N; i++) {
	          for(int j=0; j<N; j++) {
// 	#pragma HLS PIPELINE
	               Abuf[i][j] = A[i*N + j];
//	               Aibuf[i][j] = Ai[i*M + j];
	               Bbuf[i][j] = B[i*N + j];
//	               Bibuf[i][j] = Bi[i*M + j];

	          }
	     }

	 for(int i=0; i<N; i++) {
	 	          for(int j=0; j<N; j++) {
// 	 	#pragma HLS PIPELINE
	 	        	  C[i*N+j]=float(Abuf[i][j] + Bbuf[i][j]);
	 	        	  C[i*N+j]*=-1;
//	 	        	  Ci[i*M+j]=float(Aibuf[i][j] - Bibuf[i][j]);

	 	      }
	     }


}
