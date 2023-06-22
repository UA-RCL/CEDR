#ifndef __WIFI_LITE_LIB__
#define __WIFI_LITE_LIB__

struct comp_float {
   float real;
   float imag;
};
typedef struct comp_float comp_t;

float compMultReal(float aR, float aI, float bR, float bI);
float compMultImag(float aR, float aI, float bR, float bI);
float compMag(float aR, float aI);

#endif

