#define _GNU_SOURCE
#include "CyclicPrefix.h"

#include <sched.h>
#include <stdio.h>

#include "common.h"

#ifndef THREAD_PER_TASK
void cyclicPrefix(comp_t iData[], comp_t oData[], int len, int preLen) {
#else
void *cyclicPrefix(void *input) {
	comp_t *iData = ((struct args_cyclic_prefix *)input)->iData;
	comp_t *oData = ((struct args_cyclic_prefix *)input)->oData;
	int len = ((struct args_cyclic_prefix *)input)->len;
	int preLen = ((struct args_cyclic_prefix *)input)->preLen;
#endif

	int i;
	int tailIndex;

#ifdef DISPLAY_CPU_ASSIGNMENT
	printf("[INFO] TX-CRC assigned to CPU: %d\n", sched_getcpu());
#endif

	// compute tail index
	tailIndex = len - preLen;

	// copy tail part first
	for (i = 0; i < preLen; i++) oData[i] = iData[tailIndex + i];

	// copy remaining part
	for (i = 0; i < len; i++) oData[preLen + i] = iData[i];

	/*
	  printf("after cyclic prefix insertion:\n");
	  for(i=0; i<len+preLen; i++) {
	     printf("%+4.4f,%+4.4f ", oData[i].real, oData[i].imag);
	     if(i%4 == 3) printf("\n");
	  }
	  printf("\n");
	*/
}
