#ifndef __DETECION_H__
#define __DETECION_H__

#include "baseband_lib.h"

int frameDetection();
void payloadExt(comp_t dbuf[]);
void create_frameDetection(int uspRate);
void delete_frameDetection();

#endif
