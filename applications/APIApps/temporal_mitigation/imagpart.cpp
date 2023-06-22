#include "inverse.h"

void imagpart(double A[N * N], double B[N * N], double inv1[N * N], double inv2[N * N], double resultimag[N * N]) {
#pragma HLS inline region recursive
	double buffer2[N * N];
	double intmedbuff1[N * N], intmedbuff2[N * N];

	mmultiply(A, inv2, intmedbuff1);
	mmultiply(intmedbuff1, A, intmedbuff2);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			buffer2[i * N + j] = B[i * N + j] + intmedbuff2[i * N + j];
			buffer2[i * N + j] *= -1;
			resultimag[i * N + j] = buffer2[i * N + j];
			cout << buffer2[i * N + j] << " ";
		}
		cout << endl;
	}
	// display(buffer1);

	//	alternateinverse(buffer2,intmedt2);
	//	display(intmedt2);
}