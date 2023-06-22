#include "inverse.h"

#pragma SDS data mem_attribute(A : PHYSICAL_CONTIGUOUS)

void getCofactor(double A[N * N], double temp[N * N], int p, int q, int n) {
	int i = 0, j = 0;

	// Looping for each element of the matrix
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			//  Copying into temporary matrix only those element
			//  which are not in given row and column
			if (row != p && col != q) {
				j++;
				temp[i * n + j] = A[row * n + col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1) {
					j = 0;
					i++;
				}
			}
		}
	}
}
