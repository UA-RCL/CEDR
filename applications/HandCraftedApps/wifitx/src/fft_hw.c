#include <stdio.h>

#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include "baseband_lib.h"
//#include "lib.h"

static volatile uint32_t *amcor[4];

scomp_t pilot[SFT_PLT_NUM][PLT_LEN + MC_COPI];

FILE *mcorFp;
void fft_hw(comp_t fdata[], int n){
	



}
void amcor_sys_read_sequences() {

    FILE *fp;
    char buf[1024], lbuf[256];
    char *bp;
    int i, j, k;
    float dat;
    float scale = 64;

    // clear sequence buffer
    for(j=0; j<SFT_PLT_NUM; j++) {
        for(k=0; k<(PLT_LEN + MC_COPI); k++) {
            pilot[j][k].r = 0;
            pilot[j][k].i = 0;
        }
    }

    sprintf(lbuf, "../param/ppilot/ant%d.txt", 1);
    fp = fopen(lbuf, "r");
    if(fp == NULL) {
        printf("fail to open positioning pilot file, %s\n", lbuf);
        exit(1);
    }

    for(k=0; k<PLT_LEN; k++) {
        fgets(buf, 1024, fp);
        bp = buf;
        for(j=0; j<SFT_PLT_NUM; j++) {
            sscanf(bp, "%f", &dat);
            pilot[j][k].r = (short int)(round(dat));
            while(*bp == 0x20) bp++;
            while(*bp != 0x20) bp++;
            sscanf(bp, "%f", &dat);
            pilot[j][k].i = (short int)(round(dat));
            while(*bp == 0x20) bp++;
            while(*bp != 0x20) bp++;
        }
    }
    for( ; k<PLT_LEN + MC_COPI; k++) {
        for(j=0; j<SFT_PLT_NUM; j++) {
            pilot[j][k].r = 0;
            pilot[j][k].i = 0;
        }
    }

    fclose(fp);

}

void amcor_sys_load_pilot() {

    int i, j, k, offset;

    // load pilot data to the accelerator
    for(i=0; i<4; i++) {
        for(j=0, offset=0; j<25; j++, offset+=0x1000) {
            for(k=0; k<PLT_LEN+MC_COPI; k++) {
                MC_CEF(amcor[i], offset+k) = *((unsigned int *)(&pilot[j][k]));
            }
        }
    }

}


int amcor_sys_init() {

    int i;
    FILE *fp;
    scomp_t rdat;

    // map register and buffer space
    amcor[0] = (uint32_t *)memory_map(MC_BASE_ADDR_0, 0x100000);
    amcor[1] = (uint32_t *)memory_map(MC_BASE_ADDR_1, 0x100000);
    amcor[2] = (uint32_t *)memory_map(MC_BASE_ADDR_2, 0x100000);
    amcor[3] = (uint32_t *)memory_map(MC_BASE_ADDR_3, 0x100000);


    // read pilot sequence
    amcor_sys_read_sequences();

    // load pilot seuqnece onto coef mem
    amcor_sys_load_pilot();

    // init control registers
    for(i=0; i<MODULE_NUM; i++) {
        MC_REG_CTRL(amcor[i]) = MC_CTRL_STOP;
        MC_REG_STATUS(amcor[i]) = MC_STATE_CLEAR;
    }
}

void amcor_sys_load_rxpilot(int mId, int idx, scomp_t rxpilot[]) {

    int i;

    switch(idx) {
        case 0 : for(i=0; i<PLT_LEN; i++) MC_DM0(amcor[mId], i) = *((unsigned int *)(&rxpilot[i]));
                 break;
        case 1 : for(i=0; i<PLT_LEN; i++) MC_DM1(amcor[mId], i) = *((unsigned int *)(&rxpilot[i]));
                 break;
        case 2 : for(i=0; i<PLT_LEN; i++) MC_DM2(amcor[mId], i) = *((unsigned int *)(&rxpilot[i]));
                 break;
        case 3 : for(i=0; i<PLT_LEN; i++) MC_DM3(amcor[mId], i) = *((unsigned int *)(&rxpilot[i]));
                 break;
    }
}

void amcor_sys_correlation(int mId) {
    
    int i;

    MC_REG_CTRL(amcor[mId]) = MC_CTRL_START;
}

int amcor_sys_is_done(int mId) { 

    unsigned int ret;

    ret = (unsigned int)MC_REG_STATUS(amcor[mId]);
    if(ret == MC_STATE_DONE) {
        MC_REG_STATUS(amcor[mId]) = MC_STATE_CLEAR;
        return 1;
    }
    else 0; 
}

void amcor_sys_rd(int mId, int idx, int corbuf[]) {

    int i, j, k;
    int memId = 0, bufId;

/*
static int count = 0;
printf("%d\n", count++);
for(i=0; i<25; i++) {
    printf("%2d: ", i);
    for(j=0; j<5; j++) {
        printf("%08X ", MC_CEF(amcor[0], i*4096+j));
    }
    printf("\n");
}
*/

    // read correlation result
    for(i=0; i<25; i++) {
        for(j=0; j<8; j++) {
            for(k=0; k<2; k++) {
                bufId = j*25*2 + i*2 + k;
                switch(idx) {
                    case 0: corbuf[bufId] = MC_AM0(amcor[mId], memId); break;
                    case 1: corbuf[bufId] = MC_AM1(amcor[mId], memId); break;
                    case 2: corbuf[bufId] = MC_AM2(amcor[mId], memId); break;
                    case 3: corbuf[bufId] = MC_AM3(amcor[mId], memId); break;
                }
                memId++;
            }
        }
    }
}
