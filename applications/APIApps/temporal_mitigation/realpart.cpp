#include "inverse.h"

void realpart(double A[N * N], double B[N * N], double inv1[N * N], double inv2[N * N], double resultreal[N * N]) {
#pragma HLS inline region recursive
	double buffer1[N * N];
	double intmedb1[N * N], intmedb2[N * N];

	mmultiply(B, inv1, intmedb1);
	mmultiply(intmedb1, B, intmedb2);

	for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount min = 4 max = 10
		for (int j = 0; j < N; j++) {
#pragma HLS loop_tripcount min = 4 max = 10
			buffer1[i * N + j] = A[i * N + j] + intmedb2[i * N + j];
			cout << buffer1[i * N + j] << " ";
			resultreal[i * N + j] = buffer1[i * N + j];
		}
		cout << endl;
	}
	// display(buffer1);

	//	alternateinverse(buffer1,intmedt1);
	//	display(intmedt1,4,4);
}
