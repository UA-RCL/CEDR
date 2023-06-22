#include "inverse.h"

float divide(float a , float b)
{
//#pragma HLS inline
float g=a/b;
//#pragma HLS RESOURCE variable=g core=FDiv
return g;

}
