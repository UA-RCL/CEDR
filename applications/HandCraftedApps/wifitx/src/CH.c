#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "common.h"
#include "viterbi.h"
#include "awgn_ch.h"
#include "rayleigh_ch.h"
#include "baseband_lib.h"
#include "baseband_lib.h"
#include "datatypeconv.h"
 
int main () {
    
    int i,j;
    FILE *fprec;
    int pilot_symlen = 32;
    int lenth_data = OUTPUT_LEN + pilot_symlen;
    int cyclicprefixlen = (lenth_data*25)/100;
    int cyclicdatalen = lenth_data + cyclicprefixlen;
    float cyclicOData[cyclicdatalen];
    
    int framelength = (PREAMBLE_LEN*2) + cyclicdatalen;
    
    struct complex tx_data[framelength];
    float ibuf [framelength];
    struct complex obuf[framelength];
    
#ifdef AWGN_channel
    
    double nt[framelength];
    double miu=0,sigma=1;
    
#endif
    
#ifdef Rayliegh_channel
    
    double xi[framelength];
    double sigma =2;
    
#endif
    
    if ((fprec = fopen("../Input/ARM_TX_out.dat","r"))== NULL)
    {
        printf("Unable to open file to read");
        exit (-1);
    }
    
    fread(tx_data, sizeof(tx_data[0]), sizeof(tx_data)/sizeof(tx_data[0]),fprec);
    
    fclose(fprec);
    
#ifdef DEBUG
    
    printf("\nsize of data:%lu\nsize of total length:%lu\n", sizeof(tx_data[0]),sizeof(tx_data)/sizeof(tx_data[0]));
    
#endif

    //format conversion
    complextofloat (framelength, tx_data, ibuf);
    
    
#ifdef AWGN_channel
    
    for(i=0;i<framelength;i++)
    {
        nt[i]=Gaussion(miu,sigma);
    }
    
    printf("\n\n");
    
    for(i=0,j=0;i<framelength;i++)
    {
        ibuf[i] = ibuf[i] + nt[i];
        
    }
    
    floattocomplex (framelength, ibuf, obuf);
    
    if ((fprec = fopen("../Input/ARM_TX_out.dat","w"))== NULL)
    {
        printf("Unable to open file");
        exit (-1);
    }
    
#ifdef DEBUG
    
    printf("\nsize of data:%lu\nsize of total length:%lu\n", sizeof(obuf[0].real),sizeof(obuf)/sizeof(obuf[0].real));
#endif
    
    fwrite(obuf, sizeof(obuf[0]), sizeof(obuf)/sizeof(obuf[0]),fprec);
    
    fclose(fprec);

    
#endif

#ifdef Rayliegh_channel
    
    randr(xi,framelength,sigma);
    
    for(i=0;i<framelength;i++)
    {
        ibuf[i] = ibuf[i] + xi[i];
    }
    
    floattocomplex (framelength, ibuf, obuf);
    
    if ((fprec = fopen("../Input/ARM_TX_out.dat","w"))== NULL)
    {
        printf("Unable to open file");
        exit (-1);
    }

#ifdef DEBUG
    printf("\nsize of data:%lu\nsize of total length:%lu\n", sizeof(obuf[0].real),sizeof(obuf)/sizeof(obuf[0].real));
    
#endif
    
    fwrite(obuf, sizeof(obuf[0]), sizeof(obuf)/sizeof(obuf[0]),fprec);
    
    fclose(fprec);

    
#endif
    
}


    
