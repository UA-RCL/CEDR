#include "inverse.h"

void msub(double A[N * M], double Ai[N * M], double B[N * M], double Bi[N * M], double C[N * M], double Ci[N * M]) {
	double Abuf[N][M], Aibuf[N][M], Bbuf[N][M], Bibuf[N][M];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE
			Abuf[i][j] = A[i * M + j];
			Aibuf[i][j] = Ai[i * M + j];
			Bbuf[i][j] = B[i * M + j];
			Bibuf[i][j] = Bi[i * M + j];
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE
			C[i * M + j] = double(Abuf[i][j] - Bbuf[i][j]);
			Ci[i * M + j] = double(Aibuf[i][j] - Bibuf[i][j]);
		}
	}
}
