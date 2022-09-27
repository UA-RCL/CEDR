#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "baseband_lib.h"

float compMultReal(float aR, float aI, float bR, float bI) { 
   return ((aR*bR) - (aI * bI)); 
}

float compMultImag(float aR, float aI, float bR, float bI) { 
   return ((aR*bI) + (aI * bR)); 
}

float compMag(float aR, float aI) { 
   return ((aR*aR) + (aI*aI)); 
}
