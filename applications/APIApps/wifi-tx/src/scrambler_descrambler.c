#include <stdio.h>
#include <time.h>
#include "common.h"

#include "scrambler_descrambler.h"
#define SEC2NANOSEC 1000000000

static char state=0x0005;
extern unsigned int *base_addr;
/*******************************************************************
Functionn Name: scrambler
Functionality: Scrambles the data (LFSR)
Description:
    input parameters:   h ---> state
                        ibuf --> input bit stream
                        obuf ---> output bit stream
                        inlen -> Input stream len
********************************************************************/

void scrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]){

    int i;
    int y2, y5;
    char h;

    h = state;
    y2 = (0x0008 & h) >> 3;
    y5 = 0x0001 & h;

    #ifndef SCRAMBLER_CE_HW

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        struct timespec start1, end1;
        float exec_time;
        printf("[INFO] Scrambler and convolution encoder running on A53\n");
        clock_gettime(CLOCK_MONOTONIC, &start1);
        #endif

        for(i=0; i<inlen; i++) obuf[i] = ibuf[i] ^ y2 ^ y5;

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        clock_gettime(CLOCK_MONOTONIC, &end1);
        exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
        printf("[INFO] TX-Scrambler execution time (ns): %f\n", exec_time);
        #endif
    
    #else

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] Scrambler and convolution encoder running on accelerator\n");
        #endif

        unsigned int *a = malloc(2 * sizeof(unsigned int));
        for(int i = 0; i < inlen / 32; i++) {
            for(int j = 0; j < 32; j++) {
                a[i] = ibuf[(i * 32) + j] << (31 - j) | a[i];
            }
        }
        unsigned int obuf_hw1;
        unsigned int obuf_hw2;
        unsigned int obuf_hw3;
        unsigned int obuf_hw4;

        //unsigned char obuf_hw[4][4];
        unsigned int obuf_hw[4];
        int done = 0;

        struct timespec start1, end1;
        float exec_time;
        //###############################################################
        //## Begin profiling of execution time
        //###############################################################
        clock_gettime(CLOCK_MONOTONIC, &start1);
        
        // Sending 64-bit data for scrambling
        base_addr[0x00] = a[0];
        base_addr[0x00] = a[1];

        // Checking for done bit
        while (done != 1) {
            done = base_addr[0x05];
        }
         
        // Receiving the data from the accelerator
    	obuf_hw[0] = base_addr[0x01];
        obuf_hw[1] = base_addr[0x02];
    	obuf_hw[2] = base_addr[0x03];
        obuf_hw[3] = base_addr[0x04];

        // byte to bit conversion
        int jj = 0;
        int symNum       = 2;
        int sym_byte_len = 8;
        char ch;
        int j, k;

        for(i = 0; i < 4; i++) {
            for (j = 0; j < 32; j++) {
                obuf[(i * 32) + j] = (obuf_hw[i] >> j ) & 0x01;
            }
        }

        for(i = 128; i < 224; i++) {
            obuf[i] = 0;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end1);
        
        exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)
            	                  - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] TX-Scrambler and convolution encoder execution time (ns): %f\n", exec_time);
        #endif

    #endif
}

/*******************************************************************
Functionn Name: Descrambler
Functionality: Descrambles the data (LFSR)
Description:
    input parameters:   h ---> state
                        ibuf --> input bit stream
                        obuf ---> output bit stream
                        inlen -> Input stream len
*********************************************************************/

//Descrambler

void descrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]){
    
    int i;
    int z1, z2;
    char h;

    h = state;
    z1 = (0x0008 & h) >> 3;
    z2 = 0x0001 & h;
//    printf("descrmbler: input data =");    
    for(i=0;i<inlen;i++) {
		obuf[i] = ibuf[i] ^ z1 ^ z2; 
//		printf(" %u,",obuf[i]);
	}
//printf("\n");
}
