#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "baseband_lib.h"
#include "datatypeconv.h"


void complextofloat (int indatlen, comp_t *ibuf, float *obuf) {
    
    
    int i,j;
    
    
    for(i=0,j=0;i<indatlen;i=i+2)
    {
        obuf[i] = ibuf[j].real;
        obuf[i+1] = ibuf[j].imag;
        j++;
    }
}


void floattocomplex (int indatlen, float *ibuf, comp_t *obuf) {
    
    int i, j;
    
    for(i=0,j=0;i<indatlen;i=i+2)
    {
        obuf[j].real = ibuf[i];
        obuf[j].imag = ibuf[i+1];
        j++;
    }
    
}