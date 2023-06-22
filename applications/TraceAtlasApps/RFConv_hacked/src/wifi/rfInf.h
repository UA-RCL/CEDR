#ifndef __RF_INF_V__
#define __RF_INF_V__

#include "baseband_lib.h"

#define OSR_1 1
#define OSR_2 2
#define OSR_3 3
#define OSR_4 4

#define DATFILE 0
#define RFCARD 1

#define INT_2BYTE 0
#define FLT_4BYTE 1

#define NO_SMPERR 0
#define DN_SMPL 1
#define UP_SMPL 2

#define UD_INTERVAL 44
#define OSRATE 1

//#define TRF_UNIT	2048
#define TRF_TIME 2
#define TRF_UNIT (15360 * TRF_TIME)

void create_rfInf(int mode, int source, int format, int crrType, int crrIntvl);
void delete_rfInf();
int rfInfRead(comp_t *buf, int maxLen);
int rfInfWrite(comp_t *buf, int len);

#endif
