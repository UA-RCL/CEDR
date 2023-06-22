#include "baseband_lib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float compMultReal(float aR, float aI, float bR, float bI) { return ((aR * bR) - (aI * bI)); }

float compMultImag(float aR, float aI, float bR, float bI) { return ((aR * bI) + (aI * bR)); }

float compMag(float aR, float aI) { return ((aR * aR) + (aI * aI)); }
