#ifndef __DATATYPECONV_H__
#define __DATATYPECONV_H__

#include "baseband_lib.h"

/* function prototypes */

void complextofloat(int indatlen, comp_t *ibuf, float *obuf);

void floattocomplex(int indatlen, float *ibuf, comp_t *obuf);

#endif
