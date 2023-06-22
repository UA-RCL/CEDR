#include "inverse.h"

void display(float *A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << A[i * n + j] << " ";
		}
		cout << endl;
	}
}
