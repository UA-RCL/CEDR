#include <stdio.h>
#include "CyclicPrefix.h"

int cyclicPrefix(comp_t iData[], comp_t oData[], int len, int preLen) {

  int i;
  int tailIndex;

  // compute tail index
  tailIndex = len - preLen;

  // copy tail part first
  for(i=0; i<preLen; i++) oData[i] = iData[tailIndex+i];
  
  // copy remaining part
  for(i=0; i<len; i++) oData[preLen + i] = iData[i];

/*
  printf("after cyclic prefix insertion:\n");
  for(i=0; i<len+preLen; i++) {
     printf("%+4.4f,%+4.4f ", oData[i].real, oData[i].imag);
     if(i%4 == 3) printf("\n");
  }
  printf("\n");
*/

  return 0;
}
