#ifndef __CYCLIC_PREFIX_H__
#define __CYCLIC_PREFIX_H__

/* function prototypes */
#include "baseband_lib.h"

/**
 * Necessary structure for pthreads
 * when sending arguments to the
 * child thread
 */
struct args_cyclic_prefix {
	int len;
	int preLen;
	comp_t *iData;
	comp_t *oData;
};

/* function prototypes */

#ifndef THREAD_PER_TASK
void cyclicPrefix(comp_t iData[], comp_t oData[], int len, int preLen);
#else
void *cyclicPrefix(void *input);
#endif

#endif
