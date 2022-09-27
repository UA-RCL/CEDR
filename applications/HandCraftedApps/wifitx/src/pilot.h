#ifndef __PILOT_H__
#define __PILOT_H__

void pilotInsertion(comp_t idata[], comp_t odata[]);
void pilotExtract(comp_t idata[], float pilot_data[]);
void pilotRemove(int len, comp_t idata[], comp_t odata[]);

#endif
