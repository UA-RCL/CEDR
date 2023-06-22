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

//############################
//## FFT
//############################
extern TYPE         *udmabuf1_base_addr;
extern int           dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int           fd_udmabuf1;
extern int           fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern TYPE         *udmabuf2_base_addr;
extern int           dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int           fd_udmabuf2;
extern int           fft2_control_fd;
extern unsigned int *fft2_control_base_addr;
extern unsigned int  udmabuf1_phys_addr;
extern unsigned int  udmabuf2_phys_addr;

TYPE                *udmabuf1_base_addr;
int                  dma1_control_fd;
unsigned int        *dma1_control_base_addr;
int                  fd_udmabuf1;
int                  fft1_control_fd;
unsigned int        *fft1_control_base_addr;
TYPE                *udmabuf2_base_addr;
int                  dma2_control_fd;
unsigned int        *dma2_control_base_addr;
int                  fd_udmabuf2;
int                  fft2_control_fd;
unsigned int        *fft2_control_base_addr;
unsigned int         udmabuf1_phys_addr;
unsigned int         udmabuf2_phys_addr;

static char         attr[1024];
int                 fft_id;

void random_wait_time(int random_wait_time_in_us) {
    for(int k = 0; k < random_wait_time_in_us; k++) {
        for(int i = 0; i < 170; i++);
    }
}

#include "fft_hwa.h"
#include "dma.h"

#define SEC2NANOSEC 1000000000

int txOption;
int hw_fft_busy;

int fft_a53_count = 0;
int fft_acc_count = 0;

unsigned int *base_addr;

void readConfig() {

   FILE *cfp;
   char buf[1024];

   cfp = fopen("tx.cfg", "r");
   if(cfp == NULL) {
      printf("fail to open config file\n");
      exit(1);
   }

   fgets(buf, 1024, cfp);
   sscanf(buf, "%d", &txOption);
   printf("- %s\n", (txOption == 0) ? "Tx fixed string" : "Tx variable string");
}

int main() {
    
   char rate = PUNC_RATE_1_2;
   int encoderId;
   int frameLength;
   unsigned char inbit[1024];
   unsigned char scram[1024];
   signed char enc_out[OUTPUT_LEN];
   signed char enc_dep_out[OUTPUT_LEN];
   unsigned char intl_out[OUTPUT_LEN];
   double sig_real[OUTPUT_LEN];
   double sig_img[OUTPUT_LEN];
   float in_ifft[FFT_N*2];
   comp_t *ifft_in;
   comp_t pilot_out[FFT_N];
   comp_t cyclic_out[SYM_NUM*TOTAL_LEN];
   comp_t pre_out[SYM_NUM*TOTAL_LEN + PREAMBLE_LEN + 2048];
   int i, j;
   comp_t txdata[TX_DATA_LEN];

   int user_data_len;

   FILE *cfp, *fp;
   FILE *txdata_file = fopen("txdata_1.txt", "w");
    
   #ifdef SCRAMBLER_CE_HW

       // Open device and obtain virtual address to AXI slave
       int fd = open("/dev/mem",O_RDWR|O_SYNC);
       if(fd < 0) {
         printf("Can't open /dev/mem\n");
         return 1;
       }

       // Creating the virtual memory space outside the scrambler block // Helps to avoid taking more time while accessing for the first time
       base_addr = (unsigned int*) mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0xA0002000);

       if(base_addr == NULL) {
          printf("Can't obtain memory map to AXI slave connecting to hardware...\n");
          return 1;
       }

   #endif

   #ifdef FFT1_HW
       // Virtual Address to DMA Control Slave
       init_dma1();

       init_fft1();
       config_ifft(fft1_control_base_addr);

       // Virtual Address to udmabuf Buffer
       init_udmabuf1();

   #endif

   #ifdef FFT2_HW
       // Virtual Address to DMA Control Slave
       init_dma2();

       init_fft2();

       config_ifft(fft2_control_base_addr);

       // Virtual Address to udmabuf Buffer
       init_udmabuf2();

   #endif
   
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

   while(frame_count < NUM_FRAMES) {

      // input data generation
      user_data_len = txDataGen(txOption, inbit, SYM_NUM);

      // transmitter chain
      for(i=0; i<SYM_NUM; i++) {

          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          //###############################################################
          //## Scrambler (and Convolution Encoder)
          //###############################################################
	      scrambler(USR_DAT_LEN, &inbit[i*USR_DAT_LEN], scram);

          #ifndef SCRAMBLER_CE_HW
              //###############################################################
              //## SW Convolution Encoder
              //###############################################################
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif
          
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

          #ifdef FFT_HW
              srand(clock());
              hw_fft_busy = rand() % 2;

              #ifdef FFT1_HW_ONLY
                  hw_fft_busy = 0;
                  fft_id = 1;
              #endif

              #ifdef FFT2_HW_ONLY
                  hw_fft_busy = 0;
                  fft_id = 2;
              #endif

              #if !defined(FFT1_HW_ONLY) && !defined(FFT2_HW_ONLY)
                  
                  #ifdef FFT_HW_ONLY
                      #ifdef FFT_SWITCH_INST_IF_BUSY
                          fft_id = (hw_fft_busy == 1) ? 2 : 1;
                      #else
                          fft_id = 1;
                          random_wait = rand() / (float) RAND_MAX;
                          while (random_wait < FFT_WAIT_PROBABILITY) {
                              random_wait_time_in_us = rand() % 20;
                              random_wait_time(random_wait_time_in_us);
                              random_wait = rand() / (float) RAND_MAX;
                          }
                      #endif

                      hw_fft_busy = 0;
                  #endif

              #endif

              if (hw_fft_busy == 1) fft_a53_count++;
              if (hw_fft_busy == 0) fft_acc_count++;

          #else

              hw_fft_busy = 1;
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif

          #endif

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

          cyclicPrefix(pilot_out, &cyclic_out[i*(TOTAL_LEN)], FFT_N, CYC_LEN);

          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-chain execution time (ns): %f\n", exec_time);
          #endif

          #ifdef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-CRC execution time (ns): %f\n\n", exec_time);
          #endif
  
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
      printf("- RF file dump!!\n");
      rfInfWrite(txdata, TX_DATA_LEN);
      for(i = 0; i < TX_DATA_LEN; i++){
      	fprintf(txdata_file,"%f %f\n", txdata[i].real, txdata[i].imag);
      }
      
      printf("- completion!!\n\n");
      #endif

      frame_count++;
   }
  
   fclose(txdata_file);

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

   return 0;
}
