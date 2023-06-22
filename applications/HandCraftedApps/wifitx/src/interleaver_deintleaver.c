#include<stdio.h>
#include "interleaver_deintleaver.h"
/*******************************************************************
Function Name: interleaver
Functionality: Random Interleaver
Description: Randomly interleaves the data based on the input
*********************************************************************/    


int interleaver(signed char datain[],int N, unsigned char top1[]) {

    int ii,pp;

    //interleaver permutation
    for(ii=0;ii<N;ii++) {//p1
        pp = (N/4) * (ii % 4) + (ii/4) ;
//              printf("Interleaver permutation %d %d\n", ii, pp  ) ;
        top1[ii] =  datain[pp] ;
//                printf("%d,",top1[ii] ) ;
    } //Interleaver

/*
    printf("Interleaved data:\n");
    for(ii=0; ii<N; ii++) {
        printf("%d ", top1[ii]);
        if(ii%8 == 7) printf("\n");
    }
    printf("\n");
*/

    return 0;
}


/*******************************************************************
Function Name: deinterleaver
Functionality: Random deinterleaver
Description: Randomly deinterleaves the data based on the input
*********************************************************************/


int deinterleaver( signed char data[], int N, signed char top2[]) {
        int jj,kk;
        for(jj=0;jj<N;jj++) {
                kk = 56 * jj - ( N -1 )*(56*jj/N);
//                printf("permutation2 %d %d\n", jj, kk ) ;
                top2[kk] =  data[jj] ;
//                printf("%d,",top2[jj]);
        } //Deinterleaver

return 0;
}




