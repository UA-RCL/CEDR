#include "inverse.h"

void alternateinverse(float m[16], float invOut[16]) {
	float inv[16], det;
	int i;

	float mcpy[16];
	for (i = 0; i < 16; i++) {
#pragma HLS PIPELINE
		mcpy[i] = m[i];
	}

	inv[0] = mcpy[5] * mcpy[10] * mcpy[15] - mcpy[5] * mcpy[11] * mcpy[14] - mcpy[9] * mcpy[6] * mcpy[15] +
	         mcpy[9] * mcpy[7] * mcpy[14] + mcpy[13] * mcpy[6] * mcpy[11] - mcpy[13] * mcpy[7] * mcpy[10];

	inv[4] = -mcpy[4] * mcpy[10] * mcpy[15] + mcpy[4] * mcpy[11] * mcpy[14] + mcpy[8] * mcpy[6] * mcpy[15] -
	         mcpy[8] * mcpy[7] * mcpy[14] - mcpy[12] * mcpy[6] * mcpy[11] + mcpy[12] * mcpy[7] * mcpy[10];

	inv[8] = mcpy[4] * mcpy[9] * mcpy[15] - mcpy[4] * mcpy[11] * mcpy[13] - mcpy[8] * mcpy[5] * mcpy[15] +
	         mcpy[8] * mcpy[7] * mcpy[13] + mcpy[12] * mcpy[5] * mcpy[11] - mcpy[12] * mcpy[7] * mcpy[9];

	inv[12] = -mcpy[4] * mcpy[9] * mcpy[14] + mcpy[4] * mcpy[10] * mcpy[13] + mcpy[8] * mcpy[5] * mcpy[14] -
	          mcpy[8] * mcpy[6] * mcpy[13] - mcpy[12] * mcpy[5] * mcpy[10] + mcpy[12] * mcpy[6] * mcpy[9];

	inv[1] = -mcpy[1] * mcpy[10] * mcpy[15] + mcpy[1] * mcpy[11] * mcpy[14] + mcpy[9] * mcpy[2] * mcpy[15] -
	         mcpy[9] * mcpy[3] * mcpy[14] - mcpy[13] * mcpy[2] * mcpy[11] + mcpy[13] * mcpy[3] * mcpy[10];

	inv[5] = mcpy[0] * mcpy[10] * mcpy[15] - mcpy[0] * mcpy[11] * mcpy[14] - mcpy[8] * mcpy[2] * mcpy[15] +
	         mcpy[8] * mcpy[3] * mcpy[14] + mcpy[12] * mcpy[2] * mcpy[11] - mcpy[12] * mcpy[3] * mcpy[10];

	inv[9] = -mcpy[0] * mcpy[9] * mcpy[15] + mcpy[0] * mcpy[11] * mcpy[13] + mcpy[8] * mcpy[1] * mcpy[15] -
	         mcpy[8] * mcpy[3] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[11] + mcpy[12] * mcpy[3] * mcpy[9];

	inv[13] = mcpy[0] * mcpy[9] * mcpy[14] - mcpy[0] * mcpy[10] * mcpy[13] - mcpy[8] * mcpy[1] * mcpy[14] +
	          mcpy[8] * mcpy[2] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[10] - mcpy[12] * mcpy[2] * mcpy[9];

	inv[2] = mcpy[1] * mcpy[6] * mcpy[15] - mcpy[1] * mcpy[7] * mcpy[14] - mcpy[5] * mcpy[2] * mcpy[15] +
	         mcpy[5] * mcpy[3] * mcpy[14] + mcpy[13] * mcpy[2] * mcpy[7] - mcpy[13] * mcpy[3] * mcpy[6];

	inv[6] = -mcpy[0] * mcpy[6] * mcpy[15] + mcpy[0] * mcpy[7] * mcpy[14] + mcpy[4] * mcpy[2] * mcpy[15] -
	         mcpy[4] * mcpy[3] * mcpy[14] - mcpy[12] * mcpy[2] * mcpy[7] + mcpy[12] * mcpy[3] * mcpy[6];

	inv[10] = mcpy[0] * mcpy[5] * mcpy[15] - mcpy[0] * mcpy[7] * mcpy[13] - mcpy[4] * mcpy[1] * mcpy[15] +
	          mcpy[4] * mcpy[3] * mcpy[13] + mcpy[12] * mcpy[1] * mcpy[7] - mcpy[12] * mcpy[3] * mcpy[5];

	inv[14] = -mcpy[0] * mcpy[5] * mcpy[14] + mcpy[0] * mcpy[6] * mcpy[13] + mcpy[4] * mcpy[1] * mcpy[14] -
	          mcpy[4] * mcpy[2] * mcpy[13] - mcpy[12] * mcpy[1] * mcpy[6] + mcpy[12] * mcpy[2] * mcpy[5];

	inv[3] = -mcpy[1] * mcpy[6] * mcpy[11] + mcpy[1] * mcpy[7] * mcpy[10] + mcpy[5] * mcpy[2] * mcpy[11] -
	         mcpy[5] * mcpy[3] * mcpy[10] - mcpy[9] * mcpy[2] * mcpy[7] + mcpy[9] * mcpy[3] * mcpy[6];

	inv[7] = mcpy[0] * mcpy[6] * mcpy[11] - mcpy[0] * mcpy[7] * mcpy[10] - mcpy[4] * mcpy[2] * mcpy[11] +
	         mcpy[4] * mcpy[3] * mcpy[10] + mcpy[8] * mcpy[2] * mcpy[7] - mcpy[8] * mcpy[3] * mcpy[6];

	inv[11] = -mcpy[0] * mcpy[5] * mcpy[11] + mcpy[0] * mcpy[7] * mcpy[9] + mcpy[4] * mcpy[1] * mcpy[11] -
	          mcpy[4] * mcpy[3] * mcpy[9] - mcpy[8] * mcpy[1] * mcpy[7] + mcpy[8] * mcpy[3] * mcpy[5];

	inv[15] = mcpy[0] * mcpy[5] * mcpy[10] - mcpy[0] * mcpy[6] * mcpy[9] - mcpy[4] * mcpy[1] * mcpy[10] +
	          mcpy[4] * mcpy[2] * mcpy[9] + mcpy[8] * mcpy[1] * mcpy[6] - mcpy[8] * mcpy[2] * mcpy[5];

	det = mcpy[0] * inv[0] + mcpy[1] * inv[4] + mcpy[2] * inv[8] + mcpy[3] * inv[12];

	if (det == 0) {
		cout << "/n Breaking out the determinant is 0 /n";
	} else {
		det = 1.0 / det;

		for (i = 0; i < 16; i++) invOut[i] = inv[i] * det;
	}
	// return true;
}
