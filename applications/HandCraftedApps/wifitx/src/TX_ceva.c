#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "IFFT_FFT.h"
#include "scrambler_descrambler.h"
#include "CyclicPrefix.h"
#include "Preamble_ST_LG.h"
#include "common.h"
#include "viterbi.h"
#include "baseband_lib.h"
#include "interleaver_deintleaver.h"
#include "qpsk_Mod_Demod.h"
#include "datatypeconv.h"

//CEVA
#include "tcb.h"
#include "cevacode.h"
//End CEVA
unsigned char inbit [Nbits];
unsigned char inbitr[Nbits];
unsigned char scram[Nbits];
signed char enc_out[OUTPUT_LEN];
signed char enc_dep_out[OUTPUT_LEN];
unsigned char intl_out[OUTPUT_LEN];
double sig_real[OUTPUT_LEN];
double sig_img[OUTPUT_LEN];

//Ceva
#ifdef USECEVA
short int in_ifft_ceva[OUTPUT_LEN*2];
short int out_ifft_ceva[OUTPUT_LEN*2];
#endif
//END Ceva
void txPacketGeneration(void) {
    
    int i, j, index;
    
    // tx packet generation
    for(i=0; i<CODE_BLOCK/8; i++) {
        for(j=0; j<8; j++) {
            inbit[i*8 + j] = ((0x80 & ((i+1) << j)) == 0x80) ? 1 : 0;
        }
    }
    index = CODE_BLOCK%8;
    for(j=0; j<index; j++) {
        inbit[i*8 + j] = ((0x80 & ((i+1) << j)) == 0x80) ? 1 : 0;
    }
}

int main() {
    
    struct timeval tdeqpsk1, tdeqpsk2, tscrambler1, tscrambler2, tencoder1, tencoder2, tinterleaver1, tinterleaver2,tqpsk1,tqpsk2, tpilt1, tpilt2, tifft1, tifft2, tcyc1, tcyc2,tpre1, tpre2, tTX1, tTX2;
    
    int i,j,f=0,z;
    
    char state=0x0005;
    
    char rate = PUNC_RATE_1_2;
    int encoderId;
    
    //int pilot_symlen = 8;
    int pilot_symlen = 32;
    int lenth_data = OUTPUT_LEN + pilot_symlen;
    //int position[8] = {12,13,40,41,84,85,112,113};
    int position[32] = {6,7,14,15,22,23,30,31,38,39,46,47,54,55,62,63,70,71,78,79,86,87,94,95,102,103,110,111,118,119,126,127};
    //int position[8] = {12,13,40,41,84,85,102,103};
    //float pilot_data[8] = {-1,1,1,1,1,1,1,1};
    float pilot_data[32] = {-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1};
    
    float (*x)[2], (*X)[2];
    float in_ifft[OUTPUT_LEN+(pilot_symlen*2)];
    float out_ifft[OUTPUT_LEN+(pilot_symlen*2)];
    float in1[128];
    float out1[256];
    
    x = (float (*)[2])in1;
    X = (float (*)[2])out1;
    
    
    int cyclicprefixlen = (lenth_data*25)/100;
    int cyclicdatalen = lenth_data + cyclicprefixlen;
    float cyclicOData[cyclicdatalen];

    int framelength = (PREAMBLE_LEN*2) + cyclicdatalen;
    float preOData [framelength];

    float preOData_zero[framelength*4];
    float preOData_read_zero[framelength*4];
    
    struct complex tx_output[framelength];
    
    FILE *fpre;
    FILE *file;
    
#ifdef USECEVA
	        cevadata pctx;
#endif
#ifdef USECEVA            
        ceva_init(&pctx);
#endif
    
    gettimeofday(&tTX1, NULL);
    
    //.....................TRANSMITER..........................
    
    // Input bits generation
    txPacketGeneration();

    printf("\nInput bits\n");
    
    for(i=0;i<Nbits;i++)
    {
        printf("%X,",inbit[i]);
    }
    printf("\n");

    //Writing Input stream into a file
    
    file = fopen("../Input/input.dat", "wb");
    fwrite(&inbit, sizeof(inbit), 1, file);
    fclose(file);
    
    //.......................scrambler..........................
    
    gettimeofday(&tscrambler1, NULL);
    
    scrambler(Nbits, state, inbit, scram);
    
    gettimeofday(&tscrambler2, NULL);
    
#ifdef DEBUG
    printf("Scrambler output");
    for(i=0;i<Nbits;i++)
    {
        printf("%d,",scram[i]);
    }
    printf("\n\n");
#endif
    
    printf("\nScrambler = %f seconds \n", (double) (tscrambler2.tv_usec - tscrambler1.tv_usec) / 1000000 + (double) (tscrambler2.tv_sec - tscrambler1.tv_sec));
    
    
    //........................Encoder...................
    
    gettimeofday(&tencoder1, NULL);
    
    // Initilizating Encoder
    init_viterbiEncoder();
    
    // encodder instantiation
    encoderId = get_viterbiEncoder();
    set_viterbiEncoder(encoderId);
    
    viterbi_encoding(encoderId,scram,enc_out);
    
#ifdef DEBUG
    printf("Data after encoding\n");
    
    for(i=0; i<OUTPUT_LEN; i++) {
        printf("%d,",enc_out[i]);
    }
    printf("\n \n");
#endif
    
    //pucturing 2/3 rate
    
    viterbi_puncturing(rate, enc_out, enc_dep_out);
    
    
#ifdef DEBUG
    printf("Data after Puncturing");
    
    for(i=0; i<OUTPUT_LEN; i++) {
        printf("%d,",enc_dep_out[i]);
    }
    printf("\n \n");
#endif
    
    gettimeofday(&tencoder2, NULL);
    printf("\nEncoder = %f seconds \n", (double) (tencoder2.tv_usec - tencoder1.tv_usec) /1000000 + (double) (tencoder2.tv_sec - tencoder1.tv_sec));
    
    
    //.......................interleaver........................
    
    gettimeofday(&tinterleaver1, NULL);
    
    interleaver(enc_dep_out,OUTPUT_LEN,intl_out);
    
    gettimeofday(&tinterleaver2, NULL);
    
#ifdef DEBUG
    printf("\nInterleaver output\n");
    for(i=0; i<OUTPUT_LEN; i++) {
        printf("%d,",intl_out[i]);
    }
    printf("\n\n");
#endif
    
    printf("\nInterleaver = %f seconds \n", (double) (tinterleaver2.tv_usec - tinterleaver1.tv_usec) /1000000 + (double) (tinterleaver2.tv_sec - tinterleaver1.tv_sec));
    
    
    //........................QPSK Modulation...................
    
    gettimeofday(&tqpsk1, NULL);
    
    MOD_QPSK(OUTPUT_LEN,intl_out,sig_real,sig_img,in_ifft);
    
    gettimeofday(&tqpsk2, NULL);
    
#ifdef DEBUG
    printf("\nQPSK Modulation output\n");
    for(j=0;j<OUTPUT_LEN/2;j++)
    {
      		printf("%f,%f,",sig_real[j],sig_img[j]);
    }
    printf("\n\n");
#endif
    
    printf("\nQPSK = %f seconds \n", (double) (tqpsk2.tv_usec - tqpsk1.tv_usec) /1000000 + (double) (tqpsk2.tv_sec - tqpsk1.tv_sec));
    
    
    //.......................Pilot Insertion....................
    
    gettimeofday(&tpilt1, NULL);
    
    for(i=0; i<pilot_symlen; i++) {
        for (j = lenth_data - 1; j >= position[i] - 1; j--)
            in_ifft[j+1] = in_ifft[j];
        
        in_ifft[position[i]] = pilot_data[i];
   	}
    gettimeofday(&tpilt2, NULL);
    
#ifdef DEBUG
   	printf("Data after pilot insertion \n");
    
   	for (j=0; j<lenth_data; j++)
    {
      		printf("%f,", in_ifft[j]);
    }
#endif
    
    printf("\npilot_insertion = %f seconds \n", (double) (tpilt2.tv_usec - tpilt1.tv_usec) /1000000 + (double) (tpilt2.tv_sec - tpilt1.tv_sec));
    
    //...........................IFFT.........................
#ifdef USECEVA
	        for ( i=0; i<lenth_data; i++) {
			in_ifft_ceva[i] = in_ifft[i]*1000;
		}
		gettimeofday(&tifft1, NULL);
		ceva_fft(&pctx, TASKID_ASU_IFFT64, 1, 64, in_ifft_ceva, out_ifft_ceva);
		gettimeofday(&tifft2, NULL);
		for ( i=0; i<lenth_data; i++) {
			out_ifft[i] = out_ifft_ceva[i];
		}
#else
    
    ifft_v_initialize(IFFT_N, in_ifft, x);
    
    gettimeofday(&tifft1, NULL);
    
    ifft_v(IFFT_N,X,x);
    
    gettimeofday(&tifft2, NULL);
    
    // Out Buffer handling
    ifft_v_termination(IFFT_N, X, out_ifft);
#endif
    
#ifdef DEBUG
    printf("IFFT output");
    for(i=0; i<IFFT_N*2; i++)
    {
        printf("%f,",out_ifft[i]);
    }
    printf("\n \n");
#endif
    
    printf("\n64-IFFT = %f seconds \n", (double) (tifft2.tv_usec - tifft1.tv_usec) /1000000 + (double) (tifft2.tv_sec - tifft1.tv_sec));
    
    //..........................Cyclic_Prefix....................
    
    gettimeofday(&tcyc1, NULL);
    
    cyclicPrefix(out_ifft, cyclicOData, 128, 32);
    
    gettimeofday(&tcyc2, NULL);
    
#ifdef DEBUG
    printf("Data after Cyclic Prefix");
    for(i=0; i<cyclicdatalen; i++)
    {
        printf("%f,", cyclicOData[i]);
    }
#endif
    
    printf("\nCyclicPrefix =%f seconds \n", (double) (tcyc2.tv_usec - tcyc1.tv_usec) /1000000 + (double) (tcyc2.tv_sec - tcyc1.tv_sec));
    
    
    //..........................Preamble...........................
    
    gettimeofday(&tpre1, NULL);
    
    preamble(cyclicOData, preOData, cyclicdatalen);
    
    gettimeofday(&tpre2, NULL);
    
    
#ifdef DEBUG
    printf("Data after adding preamble");
    for(i=0; i<framelength; i++)
    {
        printf("%f,", preOData[i]);
    }
#endif
    
    printf("\nPreamble = %f seconds \n", (double) (tpre2.tv_usec - tpre1.tv_usec) /1000000 + (double) (tpre2.tv_sec - tpre1.tv_sec));
    
    gettimeofday(&tTX2, NULL);
    
    printf("\nTransmitter = %f seconds \n", (double) (tTX2.tv_usec - tTX1.tv_usec) /1000000 + (double) (tTX2.tv_sec - tTX1.tv_sec));
    
    
    //....................Writing the data to file in Binary format.................
    
    
    //Adding delay after one frame
    
    for(i=0; i<2*framelength; i++)
    {
        preOData_zero[i] = 0.0000;
    }
    
    for(i=0; i<framelength; i++) {
        preOData_zero[i] = preOData[i];
    }

    file = fopen("../Input/txdata.dat", "wb");
    fwrite(&preOData_zero, sizeof(preOData_zero), 1, file);
    fclose(file);
    
    
#ifdef DEBUG
    
    file = fopen("../Input/txdata.dat", "rb");
    fread(&preOData_read_zero, sizeof(preOData_read_zero), 1, file );
   	fclose(file);

     printf("after file read framelength is %d\n", framelength);
     
     for(i=0; i<4*framelength; i++)
     {
     printf("%f ", preOData_read_zero[i]);
     }
#endif
    
    //...........format conversion - Complex Data type ................
    
    floattocomplex(framelength, preOData, tx_output);
    
#if DEBUG    
    
    for(i=0;i<framelength/2;i++)
    {
        printf("%f,%f,",tx_output[i].real,tx_output[i].imag);
    }
    
#endif

    //................Saving Complex Data into the file .........................
    
    if ((fpre = fopen("../Input/ARM_TX_out.dat","w"))== NULL)
    {
        printf("Unable to open file");
        exit (-1);
    }
    
#ifdef DEBUG
    
    printf("\nsize of data:%lu\nsize of total length:%lu\n", sizeof(tx_output[0].real),sizeof(tx_output)/sizeof(tx_output[0].real));
    
#endif
    
    fwrite(tx_output, sizeof(tx_output[0]), sizeof(tx_output)/sizeof(tx_output[0]),fpre);	
    
    fclose(fpre);
    #ifdef USECEVA
	        ceva_deinit(&pctx);
	#endif

    return 0 ;
}


