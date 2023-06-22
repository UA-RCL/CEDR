#include "inverse.h"

double divide(double a, double b) {
#pragma HLS inline
	double g = a / b;
#pragma HLS RESOURCE variable = g core = FDiv
	return g;
}
