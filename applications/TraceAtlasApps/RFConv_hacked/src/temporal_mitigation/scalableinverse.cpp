#include "inverse.h"

void scalableinverse(float m[16], float invOut[16]) {
	float inv[32], ratio, a;
	float copy[4][4], copy2[4][8];
	int i, j, k;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			copy2[i][j] = m[i * 4 + j];
		}

	for (i = 0; i < 4; i++) {
		for (j = 4; j < 8; j++) {
			if (i == (j - 4))
				copy2[i][j] = 1.00;
			else
				copy2[i][j] = 0.00;
		}
	}

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (i != j) {
				ratio = copy2[j][i] / copy2[i][i];
				for (int k = 0; k < 8; k++) {
					copy2[j][k] -= ratio * copy2[i][k];
				}
			}
		}
	}

	for (i = 0; i < 4; i++) {
		a = copy2[i][i];
		for (j = 0; j < 8; j++) {
			copy2[i][j] /= a;
		}
	}

	int buffer;
	for (i = 0; i < 4; i++)
		for (j = 4; j < 8; j++) {
			buffer = j - 4;
			copy[i][buffer] = copy2[i][j];
		}

	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++) invOut[i * 4 + j] = copy[i][j];
}
