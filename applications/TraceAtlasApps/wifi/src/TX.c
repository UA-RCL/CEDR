#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include "common.h"
#include "IFFT_FFT.h"
#include "scrambler_descrambler.h"
#include "CyclicPrefix.h"
#include "Preamble_ST_LG.h"
#include "viterbi.h"
#include "baseband_lib.h"
#include "interleaver_deintleaver.h"
#include "qpsk_Mod_Demod.h"
#include "datatypeconv.h"
#include "baseband_lib.h"
#include "pilot.h"
#include "fft_hs.h"
#include "txData.h"
#include "rf_interface.h"
#include "rfInf.h"

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

static char         attr[1024];
int                 fft_id;

void random_wait_time(int random_wait_time_in_us);

#define SEC2NANOSEC 1000000000

int txOption;
int hw_fft_busy;

int fft_a53_count = 0;
int fft_acc_count = 0;

unsigned int *base_addr;

void readConfig();

int main() {
    
   char rate = PUNC_RATE_1_2;
   int encoderId;
   int frameLength;
   //unsigned char inbit[1024];
   unsigned char* inbit;
   //unsigned char scram[1024];
   unsigned char* scram;
   //signed char enc_out[OUTPUT_LEN];
   signed char* enc_out;
   //signed char enc_dep_out[OUTPUT_LEN];
   signed char* enc_dep_out;
   //unsigned char intl_out[OUTPUT_LEN];
   unsigned char* intl_out;
   //double sig_real[OUTPUT_LEN];
   double* sig_real;
   //double sig_img[OUTPUT_LEN];
   double* sig_img;
   //float in_ifft[FFT_N*2];
   comp_t *in_ifft;
   comp_t *ifft_in;
   //comp_t pilot_out[FFT_N];
   comp_t* pilot_out;
   //comp_t cyclic_out[SYM_NUM*TOTAL_LEN];
   comp_t* cyclic_out;
   //comp_t pre_out[SYM_NUM*TOTAL_LEN + PREAMBLE_LEN + 2048];
   comp_t* pre_out;
   int i, j;
   //comp_t txdata[TX_DATA_LEN];
   comp_t* txdata;

   int user_data_len;

   FILE *cfp, *fp;
   //FILE *txdata_file; // = fopen("/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/wifi/build/txdata_1.txt", "w");

   struct timespec qpr_start, qpr_end;

   for (i = 0; i < 1; i++) {}

   clock_gettime(CLOCK_MONOTONIC, &qpr_start);

   inbit = (unsigned char*) calloc(1024, sizeof(unsigned char));
   scram = (unsigned char*) calloc(1024, sizeof(unsigned char));
   enc_out = (signed char*) calloc(OUTPUT_LEN, sizeof(signed char));
   enc_dep_out = (signed char*) calloc(OUTPUT_LEN, sizeof(signed char));
   intl_out = (unsigned char*) calloc(OUTPUT_LEN, sizeof(unsigned char));
   sig_real = (double*) calloc(OUTPUT_LEN, sizeof(double));
   sig_img = (double*) calloc(OUTPUT_LEN, sizeof(double));
   in_ifft = (float*) calloc(FFT_N*2, sizeof(float));
   pilot_out = (comp_t*) calloc(FFT_N, sizeof(comp_t));
   cyclic_out = (comp_t*) calloc(SYM_NUM*TOTAL_LEN, sizeof(comp_t));
   pre_out = (comp_t*) calloc(SYM_NUM*TOTAL_LEN + PREAMBLE_LEN + 2048, sizeof(comp_t));
   txdata = (comp_t*) calloc(TX_DATA_LEN, sizeof(comp_t));
   //txdata_file = fopen("/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/wifi/build/txdata_1.txt", "w");
   
   // Object Instatiation
   init_viterbiEncoder();
   encoderId = get_viterbiEncoder();
   set_viterbiEncoder(encoderId);
   #ifdef TARGET
      create_rfInf(TXMODE, RFCARD, INT_2BYTE, NO_SMPERR, 0);
   #else
      create_rfInf(TXMODE, DATFILE, INT_2BYTE, NO_SMPERR, 0);
   #endif


   readConfig();
   struct timespec start1, end1;
   float exec_time;

   int frame_count = 0;

   //#pragma clang loop unroll(full)
   //for (int frame_count = 0; frame_count < NUM_FRAMES; frame_count++) {
//   while(frame_count < NUM_FRAMES) {
      //printf("frame_count: %d\n", frame_count);

      printf("[wifi-tx] Generating input data\n");
      // input data generation
      user_data_len = txDataGen(txOption, inbit, SYM_NUM);

      // transmitter chain
      for(i=0; i<SYM_NUM; i++) {
          printf("[wifi-tx] Encoding symbol %d of %d\n", i+1, SYM_NUM);

          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          //###############################################################
          //## Scrambler (and Convolution Encoder)
          //###############################################################
          printf("[wifi-tx] Entering scrambler\n");
	      scrambler(USR_DAT_LEN, &inbit[i*USR_DAT_LEN], scram);

          #ifndef SCRAMBLER_CE_HW
              //###############################################################
              //## SW Convolution Encoder
              //###############################################################
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif
          
              printf("[wifi-tx] Performing viterbi encoding\n");
          	  viterbi_encoding(encoderId, scram, enc_out);

              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &end1);
              exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
              printf("[INFO] TX-Encoding execution time (ns): %f\n", exec_time);a
              #endif
            
              //###############################################################
              //## SW Viterbi Puncturing
              //###############################################################
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif
          
              printf("[wifi-tx] Performing viterbi puncturing\n");
              viterbi_puncturing(rate, enc_out, enc_dep_out);

              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &end1);
              exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
              printf("[INFO] TX-Puncturing execution time (ns): %f\n", exec_time);
              #endif

          #endif  

          //###############################################################
          //## Interleaver
          //###############################################################
          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          printf("[wifi-tx] Entering interleaver\n");
          #ifdef SCRAMBLER_CE_HW
              interleaver(scram, OUTPUT_LEN, intl_out);
          #else
	          interleaver(enc_dep_out, OUTPUT_LEN, intl_out);
          #endif

          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-Interleaver execution time (ns): %f\n", exec_time);
          #endif
  
          //###############################################################
          //## Begin profiling of execution time
          //###############################################################
          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          printf("[wifi-tx] Performing QPSK Modulation\n");
	      MOD_QPSK(OUTPUT_LEN, intl_out, sig_real, sig_img, in_ifft);

          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-QPSK execution time (ns): %f\n", exec_time);
          #endif
  
          ifft_in = (comp_t *)in_ifft;

          //###############################################################
          //## Pilot Insertion
          //###############################################################
          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          printf("[wifi-tx] Inserting pilot\n");
	      pilotInsertion(ifft_in, pilot_out);

          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-Pilot execution time (ns): %f\n", exec_time);
          #endif
  
          //###############################################################
          //## Inverse FFT
          //###############################################################

          int random_wait_time_in_us; // = 15;
          float random_wait;
          //clock_gettime(CLOCK_MONOTONIC, &start1);
          //random_wait_time(random_wait_time_in_us);
          //clock_gettime(CLOCK_MONOTONIC, &end1);

          //exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          //printf("[INFO] Sleep execution time (ns): %f\n", exec_time);

          fft_id = 1;

              hw_fft_busy = 1;
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif
          
              printf("[wifi-tx] Performing IFFT\n");
	      ifft_hs(fft_id, pilot_out, FFT_N, hw_fft_busy);
          
          #ifndef FFT_HW
              
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &end1);
              exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
              printf("[INFO] TX-IFFT execution time (ns): %f\n", exec_time);
              #endif

          #endif

          //###############################################################
          //## CRC
          //###############################################################
          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          printf("[wifi-tx] Computing CRC\n");
          cyclicPrefix(pilot_out, &cyclic_out[i*(TOTAL_LEN)], FFT_N, CYC_LEN);

          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          //printf("[INFO] TX-chain execution time (ns): %f\n", exec_time);
          #endif

          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-CRC execution time (ns): %f\n\n", exec_time);
          #endif
  
          printf("[wifi-tx] Finished symbol %d\n\n", i+1);
      }
   
      // zero padding
      for(i=0; i<256; i++) { // 512 zero pad
         pre_out[i].real = pre_out[i].imag = 0;
      }
      // data payload
      preamble(cyclic_out, &pre_out[256], SYM_NUM*(TOTAL_LEN)); // 322 preamble + SYM_NUM*80
      // total = 256 + 322 + SYM_NUM*(64+16)
  
      // frame duplications
      frameLength = 256 + PREAMBLE_LEN + SYM_NUM*(TOTAL_LEN);//in complex number
      for(i=0; i<TX_DATA_LEN - frameLength; i+=frameLength) {
         for(j=0; j<frameLength; j++) {
            txdata[i+j].real = pre_out[j].real;
            txdata[i+j].imag = pre_out[j].imag;
         }
      }

      for( ; i<TX_DATA_LEN; i++) {
         txdata[i].real = 0;
         txdata[i].imag = 0;
      }
  
      // send data
      //txDataDump(pre_out, SYM_NUM*(64+16)+322);

      #ifdef TARGET
      if(txOption == 0) {
         printf("press enter to transmit: ");
         getchar();
      }
      for(i=0; i<3; i++) {
         rfInfWrite(txdata, TX_DATA_LEN);
      }
      #else
      //printf("- RF file dump!!\n");
      //rfInfWrite(txdata, TX_DATA_LEN);
      //for(i = 0; i < TX_DATA_LEN; i++){
      //	fprintf(txdata_file,"%f %f\n", txdata[i].real, txdata[i].imag);
      //}
      
      //printf("- completion!!\n\n");
      #endif

      //frame_count++;
   //}
  
   //fclose(txdata_file);

   #ifdef FFT_HW
       printf("[INFO] Count of IFFT on A53: %d\n", fft_a53_count);
       printf("[INFO] Count of IFFT on ACC: %d\n\n\n", fft_acc_count);
   #endif

   #ifdef FFT1_HW
       close_dma1();
       close_fft1();
       munmap(udmabuf1_base_addr, 8192);
       close(fd_udmabuf1);
   #endif

   #ifdef FFT2_HW
       close_dma2();
       close_fft2();
       munmap(udmabuf2_base_addr, 8192);
       close(fd_udmabuf2);
   #endif

   clock_gettime(CLOCK_MONOTONIC, &qpr_end);
   exec_time = ((double)qpr_end.tv_sec*SEC2NANOSEC + (double)qpr_end.tv_nsec) - ((double)qpr_start.tv_sec*SEC2NANOSEC + (double)qpr_start.tv_nsec);
   //printf("WiFi TX chain complete (took %f seconds)\n", exec_time);
   printf("[wifi-tx] WiFi TX chain complete\n", exec_time);

   return 0;
}

void random_wait_time(int random_wait_time_in_us) {
    for(int k = 0; k < random_wait_time_in_us; k++) {
        for(int i = 0; i < 170; i++);
    }
}

void readConfig() {

    FILE *cfp;
    char buf[1024];

    //cfp = fopen("/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/wifi/build/tx.cfg", "r");
    cfp = fopen("./input/tx.cfg", "r");
    if(cfp == NULL) {
        char currWorkingDir[PATH_MAX];
        printf("fail to open config file: %s\n", strerror(errno));
        getcwd(currWorkingDir, PATH_MAX);
        printf("current working dir: %s\n", currWorkingDir);
        exit(1);
    }

    fgets(buf, 1024, cfp);
    sscanf(buf, "%d", &txOption);
    printf("- %s\n", (txOption == 0) ? "Tx fixed string" : "Tx variable string");
}
