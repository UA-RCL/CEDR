#define _GNU_SOURCE
#include<stdio.h>
#include <sched.h>

#include "common.h"
#include "interleaver_deintleaver.h"
/*******************************************************************
Function Name: interleaver
Functionality: Random Interleaver
Description: Randomly interleaves the data based on the input
*********************************************************************/    

#ifndef THREAD_PER_TASK
void interleaver(signed char datain[],int N, unsigned char top1[]) {
#else
void * interleaver(void *input) {

    unsigned char *datain = ((struct args_interleaver*)input)->datain; 
    int N = ((struct args_interleaver*)input)->N; 
    unsigned char *top1 = ((struct args_interleaver*)input)->top1;
#endif 

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] TX-Interleaver assigned to CPU: %d\n", sched_getcpu());
    #endif

    int ii,pp;

    //interleaver permutation
    for(ii=0;ii<N;ii++) {//p1
        pp = (N/4) * (ii % 4) + (ii/4) ;
        top1[ii] =  datain[pp] ;
    } //Interleaver

}


/*******************************************************************
Function Name: deinterleaver
Functionality: Random deinterleaver
Description: Randomly deinterleaves the data based on the input
*********************************************************************/


#ifndef THREAD_PER_TASK
int deinterleaver( signed char data[], int N, signed char top2[]) {
#else
void * deinterleaver(void *input) {

    signed char *data = ((struct args_deinterleaver*)input)->datain;
    int N = ((struct args_deinterleaver*)input)->N;
    signed char *top2 = ((struct args_deinterleaver*)input)->top2;
#endif
    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] RX-Deinterleaver assigned to CPU: %d\n", sched_getcpu());
    #endif

        int jj,kk;
        for(jj=0;jj<N;jj++) {
                kk = 56 * jj - ( N -1 )*(56*jj/N);
                top2[kk] =  data[jj] ;
        } //Deinterleaver

}




