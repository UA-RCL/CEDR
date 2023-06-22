#include "inverse.h"

void msub(float A[N * M], float Ai[N * M], float B[N * M], float Bi[N * M], float C[N * M], float Ci[N * M]) {
	float Abuf[N][M], Aibuf[N][M], Bbuf[N][M], Bibuf[N][M];

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
			C[i * M + j] = float(Abuf[i][j] - Bbuf[i][j]);
			Ci[i * M + j] = float(Aibuf[i][j] - Bibuf[i][j]);
		}
	}
}
