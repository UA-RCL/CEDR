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
#include "awgn_ch.h"
#include "rayleigh_ch.h"
#include "ch_Est_Equa.h"
#include "detection.h"
#include "rt_nonfinite.h"
#include "channel_Eq.h"
#include "channel_Eq_terminate.h"
#include "channel_Eq_initialize.h"
#include "decode.h"
#include "datatypeconv.h"
//CEVA
#include "tcb.h"
#include "cevacode.h"
//End CEVA

unsigned char descram[Nbits];
signed char dec_in[OUTPUT_LEN];
unsigned char dec_out[Nbits];
signed char dec_pun_out[OUTPUT_LEN];
signed char deintl_out[OUTPUT_LEN];
double out_real[OUTPUT_LEN];
double out_img[OUTPUT_LEN];
signed char outbit[OUTPUT_LEN];

//Ceva
#ifdef USECEVA
short int in_fft_ceva[OUTPUT_LEN*2];
short int out_fft_ceva[OUTPUT_LEN*2];
float in_vit_ceva[OUTPUT_LEN];
int dec_out_ceva[Nbits];
#endif
//END Ceva

int main() {
    
    struct timeval tdeqpsk1, tdeqpsk2, tdescram1, tdescram2, trpilt1, trpilt2, tdecoder1, tdecoder2, tdeinterleaver1, tdeinterleaver2, tfft1, tfft2, tTX1, tTX2, tRX1, tRX2;
    
    int i,j,f=0,z;
    
    FILE *fprer;
    
    char state=0x0005;

    int pilot_symlen = 32;
    int lenth_data = OUTPUT_LEN + pilot_symlen;
    
    int cyclicprefixlen = (lenth_data*25)/100;
    int cyclicdatalen = lenth_data + cyclicprefixlen;
    float cyclicOData[cyclicdatalen];
    int framelength = (PREAMBLE_LEN*2) + cyclicdatalen;

    struct complex rx_data_N1[framelength];
    float rx_data_N2[framelength];
    float rx_data_N3[framelength];
    
    struct complex out_fd[OUTPUT_LEN+(pilot_symlen*2)];
    
    //int position[8] = {12,13,40,41,84,85,112,113};
    int position[32] = {6,7,14,15,22,23,30,31,38,39,46,47,54,55,62,63,70,71,78,79,86,87,94,95,102,103,110,111,118,119,126,127};
    //int position[8] = {12,13,40,41,84,85,102,103};
    //float pilot_data[8] = {-1,1,1,1,1,1,1,1};
    float pilot_data[32] = {-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1};
    float pilotdata_rx[pilot_symlen];
    
    float (*x1)[2], (*X1)[2];
    float in3[128];
    float out3[256];
    
    x1 = (float (*)[2])in3;
   	X1 = (float (*)[2])out3;

    float in_fft[OUTPUT_LEN+(pilot_symlen*2)];
    float out_fft[OUTPUT_LEN+(pilot_symlen*2)];
    float out_fft1[OUTPUT_LEN+(pilot_symlen*2)];
    
    creal_T tpilot[16];
    creal_T rpilot[16];
    creal_T fftout[64];
    creal_T dftm[4096];
    creal_T eqData[64];
    
    float dftm_r[64*64*2];

    float out_fft_c2f[OUTPUT_LEN+(pilot_symlen*2)];
    
    int decoderId;
    char rate = PUNC_RATE_1_2;

    unsigned char inbitr[Nbits];
    int wrngbit=0;
    float Ber;
    int InOutCom[Nbits];
    
    FILE *file;
    
    
    for(i=0;i<16;i++)
    {
        tpilot[i].re = 0;
        tpilot[i].im = 0;
        rpilot[i].re =0;
        rpilot[i].im =0;
    }
    
    for(i=0;i<32;i++)
    {
        fftout[i].re =0;
        fftout[i].im =0;
        eqData[i].re =0;
        eqData[i].im=0;
        
    }
    
    
#ifdef FILE_TX_RX
    
    if ((fprer = fopen("../Input/ARM_TX_out.dat","r"))== NULL)
    {
        printf("Unable to open file to read");
        exit (-1);
    }
    
    fread(rx_data_N1, sizeof(rx_data_N1[0]), sizeof(rx_data_N1)/sizeof(rx_data_N1[0]),fprer);
    
    fclose(fprer);
    
    //Format conversion
    
    complextofloat(framelength, rx_data_N1, rx_data_N2);
    
    
#ifdef DEBUG
    
    printf("\nsize of data:%lu\nsize of total length:%lu\n", sizeof(rx_data_N1[0]),sizeof(rx_data_N1)/sizeof(rx_data_N1[0]));
    
#endif

#endif 

#ifdef WIRELESS
    
    //............................Frame Detection...............
    
    // reserve resources
    create_frameDetection();
    
    // frame detection
    
   	frameDetection(out_fd);
    
    //........................Format Conversion.................
    
    complextofloat(FFT_N, out_fd, in_fft);
    
#endif
    
    gettimeofday(&tRX1, NULL);
    
    //...........................FFT......................
    
    
#ifdef WIRELESS
#ifdef USECEVA
	for(i=0;i<lenth_data;i++) {
		in_fft_ceva[i] = in_fft[i]*1000;
	}
	ceva_fft(&pctx,TASKID_ASU_FFT64, 1, 64, in_fft_ceva, out_fft_ceva);
	for(i=0; i<FFT_N*2; i++) {
		out_fft_ceva[i] = out_fft_ceva[i]/4;
		out_fft[i] = (float)out_fft_ceva[i]/(float)1000;
	}
#else
    ifft_v_initialize(IFFT_N, in_fft, x1);
    
    gettimeofday(&tfft1, NULL);
    fft_v(FFT_N,x1,X1);
    gettimeofday(&tfft2, NULL);
    
    ifft_v_termination(IFFT_N, X1, out_fft);
#endif
    
#endif
    
#ifdef FILE_TX_RX
    
    printf("\nPaylaod extraction \n");
    
    for(i=(PREAMBLE_LEN*2)+cyclicprefixlen,j=0; i<framelength; i++)
    {
        rx_data_N3[j] = rx_data_N2[i];
        printf("%f,",rx_data_N3[j]);
        j++;
    }
    
    ifft_v_initialize(IFFT_N, rx_data_N3, x1);
    
    gettimeofday(&tfft1, NULL);
    fft_v(FFT_N, x1, X1);
    gettimeofday(&tfft2, NULL);
    
    ifft_v_termination(IFFT_N, X1, out_fft);
    
#endif    
    
#ifdef DEBUG
    printf("FFT Output");
    
    for(i=0; i<FFT_N*2; i++)
    {
        printf("%f,",out_fft[i]);
    }
    printf("\n\n");
#endif
    
    printf("\n64-FFT = %f seconds \n", (double) (tfft2.tv_usec - tfft1.tv_usec) /1000000 + (double) (tfft2.tv_sec - tfft1.tv_sec));
    
    
    //..........................pilot data at RX.............
    
    for(i=0; i<pilot_symlen; i++)
    {
        pilotdata_rx[i] = out_fft[position[i]];
    }
    
#ifdef DEBUG
    
    printf("\nPilot Data at receiver\n");
    for(i=0; i<pilot_symlen; i++)
    {
        printf("%f,",pilotdata_rx[i]);
    }
    printf("\n\n");
#endif
    
    //........................Channel Estimation and Equalization.............
    
    printf("\nTXpilot\n");
    
    // Txpilot
    f2c(pilot_symlen, pilot_data, tpilot);
    
    //RxPilot

    f2c(pilot_symlen, pilotdata_rx, rpilot);

    //FFT Data
    
    f2c(FFT_N*2, out_fft, fftout);
    
    //DFTmatrix
    
    file = fopen("../Input/F_complex.dat", "rb");
    fread(&dftm_r, sizeof(dftm_r), 1, file);
    fclose(file);
    
  
    for(i=0,j=0; i<4096; i++)
    {
        dftm[j].re = dftm_r[i];
        dftm[i].im = dftm_r[i+4096];
        j++;
    }
   
    printf("\n\n");
    
    channel_Eq(tpilot,rpilot,fftout,dftm,eqData);
    
    
    //Conversion Complex to float
    
    c2f(FFT_N, eqData, out_fft_c2f);
    
    //..........................Remove pilot insertion.............
    
    gettimeofday(&trpilt1, NULL);
    
    for(i=0; i<pilot_symlen; i++)
   	{
        j = position[i]-i;
        for (; j < lenth_data - 1 ; j++ )
        {
            //EqRxData[j] = EqRxData[j+1];
            out_fft_c2f[j] = out_fft_c2f[j+1];
        }
    }
    
    gettimeofday(&trpilt2, NULL);
    
#ifdef DEBUG
    printf("\nData after pilot removal\n");
    for(i=0;i<INPUT_LEN*2;i++)
    {
        printf("%f,",out_fft_c2f[i]);
    }
    printf("\n\n");
#endif
    
    printf("\nPilot data removal = %f seconds \n", (double) (trpilt2.tv_usec - trpilt1.tv_usec) /1000000 + (double) (trpilt2.tv_sec - trpilt1.tv_sec));
    
    
    //...........................Demod-QPSK........................
    
    
    gettimeofday(&tdeqpsk1, NULL);
    
    DeMOD_QPSK(INPUT_LEN,out_fft_c2f,outbit);
    
    gettimeofday(&tdeqpsk2, NULL);
    
#ifdef DEBUG
    printf("QPSK Demodulated output");
    for(i=0;i<INPUT_LEN*2;i++)
    {
        printf("%d,",outbit[i]);
    }
    printf("\n\n");
#endif
    
    printf("\nDeQpsk = %f seconds \n", (double) (tdeqpsk2.tv_usec - tdeqpsk1.tv_usec) /
           1000000 + (double) (tdeqpsk2.tv_sec - tdeqpsk1.tv_sec));
    
    //.........................Deinterleaver......................
    
    gettimeofday(&tdeinterleaver1, NULL);
    
    deinterleaver(outbit,OUTPUT_LEN,deintl_out);
    
    gettimeofday(&tdeinterleaver2, NULL);
    
#ifdef DEBUG
    printf("Deinterleaver output\n");
    
    for(i=0; i<OUTPUT_LEN; i++) {
        printf("%d,",deintl_out[i]);
    }
    printf("\n\n");
#endif
    
    printf("\nDeinterleaver = %f seconds \n", (double) (tdeinterleaver2.tv_usec - tdeinterleaver1.tv_usec) /1000000  + (double) (tdeinterleaver2.tv_sec - tdeinterleaver1.tv_sec));
    
    //...........................decoder........................
    
    gettimeofday(&tdecoder1, NULL);
   
    // decoder instantiation
    
    init_viterbiDecoder();
    decoderId = get_viterbiDecoder();
    set_viterbiDecoder(decoderId);
    
    //format conversion
    
#ifdef HARDINPUT
    
    formatConversion(rate,deintl_out,dec_in);
    
#endif
    
    // depuncturing
    
#ifdef HARDINPUT
    
    viterbi_depuncturing(rate, dec_in, dec_pun_out);
#else
    viterbi_depuncturing(rate, deintl_out, dec_pun_out);
#endif

#ifdef USECEVA 
	for (i=0; i<OUTPUT_LEN; i++) {
		in_vit_ceva[i] = (float) dec_pun_out[i];
	}
	ceva_fft(&pctx, TASKID_ASU_VITERBIK7, 1 , OUTPUT_LEN, in_vit_ceva, dec_out_ceva);
	for(i=0; i<Nbits; i++) {
		dec_out[i] = (unsigned char) dec_out_ceva[i];
	}
#else
	viterbi_decoding(decoderId, dec_pun_out, dec_out);
#endif    
    gettimeofday(&tdecoder2, NULL);
    

#ifdef DEBUG
    printf("\nrxData before decoding");
    for(i=0; i<OUTPUT_LEN; i++)
    {
        if(i%8==0) printf("\n");
        printf("%d,",deintl_out[i]);
    }
    printf("\n\n");
#endif
    
#ifdef DEBUG
    printf("Data after decoding\n");
    for(i=0; i<=Nbits; i++) {
        printf("%d,",dec_out[i]);
    }
#endif
    
    printf("\nDecoder = %f seconds \n", (double) (tdecoder2.tv_usec - tdecoder1.tv_usec) /1000000 + (double) (tdecoder2.tv_sec - tdecoder1.tv_sec));
    
     
    //.......................Descrambler........................
    
    gettimeofday(&tdescram1, NULL);
    
    descrambler(Nbits, state, dec_out, descram);
    
    gettimeofday(&tdescram2, NULL);
    
#ifdef DEBUG
    for(i=0;i<Nbits;i++)
    {
        printf("%d,",descram[i]);
    }
#endif
    
    printf("\nDescrambler = %f seconds \n", (double) (tdescram2.tv_usec - tdescram1.tv_usec) /1000000 + (double) (tdescram2.tv_sec - tdescram1.tv_sec));
    
    gettimeofday(&tRX2, NULL);
    printf("\nReceiver = %f seconds \n", (double) (tRX2.tv_usec - tRX1.tv_usec) /1000000 + (double) (tRX1.tv_sec - tRX1.tv_sec));
    

#ifdef WIRELESS
    messagedecoder((unsigned char *)descram);
#endif
    
#ifdef FILE_TX_RX
    //.......................I/P & O/P Comparison........................\n");
   
    file = fopen("../Input/input.dat", "rb");
    fread(&inbitr, sizeof(inbitr), 1, file );
    fclose(file);
    
    printf("\nI/P & O/P Comparison\n");
    for(i=0; i<Nbits; i++)
    {
        InOutCom[i]= descram[i]-inbitr[i];
        printf("%d ",InOutCom[i]);
        if(InOutCom[i]== 1 || InOutCom[i]== -1)
        {
            ++wrngbit;
        }
        
    }
    printf("\n");
    
    //............................BER.....................................
    
#ifdef DEBUG
    printf("\nno of wrngbits:%d\n",wrngbit);
    
    Ber = wrngbit / (float) Nbits;   	
    
    printf("\nBER:%f\n\n",Ber);
#endif
    
#endif
#ifdef USECEVA
	        ceva_deinit(&pctx);
#endif    
    return 0;
    
}


    
    

