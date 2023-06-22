#include "inverse.h"

/* __attribute__((always_inline)) */ void hermitian(float S[N * M], float Si[N * M], float Shermitian[M * N], float Shermitianimag[M * N]) {
	int i, j;
	// Initializing local buffers  to store the S matrices
	//	float Sbuf[4][64],Sibuf[4][64],Stbuf[64][4],Stibuf[64][4];

	//// Transposing the matrix first
	//	cout<<"Printing the original array first \n";
	//
	//	for(i=0;i<N;i++)
	//	{
	//		for(j=0;j<M;j++)
	//		{
	//			Sbuf[i][j]=S[i*M+j];
	//			Sibuf[i][j]=Si[i*M+j];
	//		}
	//	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			Shermitian[j * N + i] = S[i * M + j];
			//		  cout<<Shermitian[j*N+i]<<" ";
			Shermitianimag[j * N + i] = -Si[i * M + j];
		}
		//		cout<<"\n";
	}

	//	cout<<"Now printing the transposed real array \n";
	//
	//	for(i=0;i<M;i++)
	//		{
	//			for(j=0;j<N;j++)
	//			{
	//				cout<<Shermitian[i*N+j]<<" ";
	//			}
	//			cout<<"\n";
	//		}

	//	// Computing transpose on the local buffer
	//	for (i=0;i<N;i++)
	//		{ for (j=0;j<M;j++)
	//			{	Stbuf[j][i]=Sbuf[i][j];
	//				cout<<Stbuf[j][i]<<" ";
	//				Stibuf[j][i]=Sibuf[i][j];
	//				Stibuf[j][i]*=-1;
	//
	//			}
	//			cout<<"\n";
	//
	//		}
	//
	//
	//	/// Now going to compute the complex conjugate for the above matrix
	//
	//	for (i=0;i<M;i++)
	//	{ for(j=0;j<N;j++)
	//		{ Shermitian[i*N+j]=Stbuf[i][j];
	//		  cout<<Shermitian[i*N+j]<<" ";
	//		  Shermitianimag[i*N+j]=Stibuf[i][j];
	//		}
	//		cout<<"\n";
	//
	//	}
	//
	//
}
