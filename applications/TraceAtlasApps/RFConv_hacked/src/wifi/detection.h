#ifndef __DETECION_H__
#define __DETECION_H__

#include "baseband_lib.h"

int frameDetection();

struct args_payload {
	comp_t *dbuf;
};

#ifndef THREAD_PER_TASK
void payloadExt(comp_t dbuf[]);
#else
void *payloadExt(void *input);
#endif

void create_frameDetection(int uspRate);
void delete_frameDetection();

#endif
