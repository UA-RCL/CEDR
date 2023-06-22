#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "common.h"
#include "baseband_lib.h"

#define PILOT_LEN 		16

static int pilotLoc[PILOT_LEN] = {7, 15, 23, 31, 39, 47, 55, 63, 
                           71, 79, 87, 95, 103, 111, 119, 127};

static comp_t pilotData[PILOT_LEN] = {{-1,1}, {1,1}, {1,1}, {1,1}, 
                                      {-1,1}, {1,1}, {1,1}, {1,1},
                                      {-1,1}, {1,1}, {1,1}, {1,1},
                                      {-1,1}, {1,1}, {1,1}, {1,1}};

void pilotInsertion(comp_t idata[], comp_t odata[]) {

    int i, j, k;

    for(i=0, j=0, k=0; i<128; i++) {
       if(pilotLoc[j] == i) {
          odata[i] = pilotData[j];
          j++;
       }
       else {
          odata[i] = idata[k];
          k++;
       }
    }

/*
    printf("after pilot insertion:\n");
    for(i=0; i<64; i++) {
       printf("%+4.4f,%+4.4f ", odata[i].real, odata[i].imag); 
       if(i%4 == 3) printf("\n");
    }
    printf("\n");
*/
}

void pilotExtract(comp_t idata[], float pilot_data[]) {

    int i, j;
    
    for(i=0, j=0; i<PILOT_LEN; i++) {
        pilot_data[j++] = idata[pilotLoc[i]].real;
        pilot_data[j++] = idata[pilotLoc[i]].imag;
    }
}

void pilotRemove(int len, comp_t idata[], comp_t odata[]) {

    int i, j, k;
    
    for(i=0, j=0, k=0; i<len; i++) {
        if(i != pilotLoc[j]) odata[k++] = idata[i];
        else j++;
    }
}
