#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include "common.h"
#include <semaphore.h>
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
#include <pthread.h>
#include <assert.h>

void papi_init();
void get_tx_ifft_data();

int elab_papi_end (const char *format, ...);

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

extern sem_t mutex;
sem_t        mutex;

static char         attr[1024];
int                 fft_id;

#include "fft_hwa.h"
#include "dma.h"

#define SEC2NANOSEC 1000000000

int txOption;
int hw_fft_busy;

unsigned int *base_addr;

void readConfig() {

   char buf[1024];

	// DASH_DATA
	if( !getenv("DASH_DATA") )
	{
		printf("in TX.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char* file0 = "Dash-RadioCorpus/wifi/tx.cfg";
	char* path0 = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
	strcat(path0, getenv("DASH_DATA"));
	strcat(path0, file0);
	FILE* cfp = fopen(path0, "r");
	free(path0);

    if(cfp == NULL) {
        printf("in TX.c:\n\tFATAL:%s was not found!\n", file0);
        exit(1);
    }

   	fgets(buf, 1024, cfp);
   	sscanf(buf, "%d", &txOption);
	fclose(cfp);
   	//printf("- %s\n", (txOption == 0) ? "Tx fixed string" : "Tx variable string");
}

void* wifi_tx() {

   int fft_a53_count = 0;
   int fft_acc_count = 0;

   float FFT_WAIT_PROBABILITY       = 0.5;
   float FFT_RUN_ON_CPU_PROBABILITY = 0.5;

   //FFT_WAIT_PROBABILITY       = fft_wait_probability;
   //FFT_RUN_ON_CPU_PROBABILITY = fft_run_on_cpu_probability;
   FFT_WAIT_PROBABILITY       = 0;
   FFT_RUN_ON_CPU_PROBABILITY = 0;
    
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

	// DASH_DATA
	if( !getenv("DASH_DATA") )
	{
		printf("in TX.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char* file1 = "Dash-RadioCorpus/wifi/txdata_1.txt";
	char* path1 = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
	strcat(path1, getenv("DASH_DATA"));
	strcat(path1, file1);
	FILE* txdata_file = fopen(path1, "w");
	free(path1);

    if(txdata_file == NULL) {
        printf("in TX.c:\n\tFATAL: %s was not found!\n", file1);
        exit(1);
    }

   // Object Instatiation
   init_viterbiEncoder();
   encoderId = get_viterbiEncoder();
   set_viterbiEncoder(encoderId);
   #ifdef TARGET
      create_rfInf(TXMODE, RFCARD, INT_2BYTE, NO_SMPERR, 0);
   #else
      create_rfInf(TXMODE, DATFILE, INT_2BYTE, NO_SMPERR, 0);
   #endif

   //readConfig();
   struct timespec start1, end1;
   float exec_time;

   int frame_count = 0;

   #ifdef ACC_RX_IFFT
       get_tx_ifft_data();
   #endif

   #ifdef THREAD_PER_TASK
   pthread_t thread_scrambler;
   pthread_t thread_encoder;
   pthread_t thread_interleaver;
   pthread_t thread_qpsk_mod;
   pthread_t thread_pilot;
   pthread_t thread_ifft;
   pthread_t thread_crc;
   
   pthread_attr_t attr_thread_scrambler;
   pthread_attr_t attr_thread_encoder;
   pthread_attr_t attr_thread_interleaver;
   pthread_attr_t attr_thread_qpsk_mod;
   pthread_attr_t attr_thread_pilot;
   pthread_attr_t attr_thread_ifft;
   pthread_attr_t attr_thread_crc;

   pthread_attr_init(&attr_thread_scrambler);
   pthread_attr_init(&attr_thread_encoder);
   pthread_attr_init(&attr_thread_interleaver);
   pthread_attr_init(&attr_thread_qpsk_mod);
   pthread_attr_init(&attr_thread_pilot);
   pthread_attr_init(&attr_thread_ifft);
   pthread_attr_init(&attr_thread_crc);

   pthread_attr_setname(&attr_thread_scrambler  , "scrambler");
   pthread_attr_setname(&attr_thread_encoder    , "encoder");
   pthread_attr_setname(&attr_thread_interleaver, "interleaver");
   pthread_attr_setname(&attr_thread_qpsk_mod   , "qpsk_mod");
   pthread_attr_setname(&attr_thread_pilot      , "pilot");
   pthread_attr_setname(&attr_thread_ifft       , "ifft");
   pthread_attr_setname(&attr_thread_crc        , "crc");
   #endif

    while(frame_count < NUM_FRAMES) {

      // input data generation
      user_data_len = txDataGen(0, inbit, SYM_NUM);

      // transmitter chain
      for(i=0; i<SYM_NUM; i++) {

          //srand(clock());
          srand(frame_count + 1);
          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          clock_gettime(CLOCK_MONOTONIC, &start1);
          #endif

          //###############################################################
          //## Scrambler (and Convolution Encoder)
          //###############################################################
          
          #ifndef THREAD_PER_TASK
          scrambler(USR_DAT_LEN, &inbit[i*USR_DAT_LEN], scram);
          #else
          /* Parse the structure that will be sent as arguments for the scrambler fx */
          struct args_scrambler *thread_param_scrambler = (struct args_scrambler *)malloc(sizeof(struct args_scrambler));
          thread_param_scrambler->inlen = USR_DAT_LEN;
          thread_param_scrambler->ibuf = &inbit[i*USR_DAT_LEN];
          thread_param_scrambler->obuf = scram;
          assert(pthread_create(&thread_scrambler, &attr_thread_scrambler, scrambler, (void *)thread_param_scrambler) == 0);
          assert (pthread_join(thread_scrambler, NULL) == 0);
          free(thread_param_scrambler);
          #endif

          #ifndef SCRAMBLER_CE_HW
              //###############################################################
              //## SW Convolution Encoder
              //###############################################################
              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &start1);
              #endif

          	  #ifndef THREAD_PER_TASK
              viterbi_encoding(encoderId, scram, enc_out);
              #else
              /* Parse the structure that will be sent as arguments for the viterbi encoder fx */
              struct args_encoder *thread_param_encoder = (struct args_encoder *)malloc(sizeof(struct args_encoder));
              thread_param_encoder->eId = encoderId;
              thread_param_encoder->iBuf = scram;
              thread_param_encoder->oBuf = enc_out;
              assert (pthread_create(&thread_encoder, &attr_thread_encoder, viterbi_encoding, (void *)thread_param_encoder) == 0);
              assert (pthread_join(thread_encoder, NULL) == 0);
              free(thread_param_encoder);
              #endif

              #ifdef PRINT_BLOCK_EXECUTION_TIMES
              clock_gettime(CLOCK_MONOTONIC, &end1);
              exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
              printf("[INFO] TX-Encoding execution time (ns): %f\n", exec_time);
              #endif
            
              //###############################################################
              //## SW Viterbi Puncturing
              //###############################################################
              //#ifdef PRINT_BLOCK_EXECUTION_TIMES
              //clock_gettime(CLOCK_MONOTONIC, &start1);
              //#endif
          
              //viterbi_puncturing(rate, enc_out, enc_dep_out);

              //#ifdef PRINT_BLOCK_EXECUTION_TIMES
              //clock_gettime(CLOCK_MONOTONIC, &end1);
              //exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
              //printf("[INFO] TX-Puncturing execution time (ns): %f\n", exec_time);
              //#endif

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
              #ifndef THREAD_PER_TASK
	          interleaver(enc_out, OUTPUT_LEN, intl_out);
              #else
                 /* Parse the structure that will be sent as arguments for the interleaver fx */
                 struct args_interleaver *thread_param_interleaver = (struct args_interleaver *)malloc(sizeof(struct args_interleaver));
                 thread_param_interleaver->N = OUTPUT_LEN;
                 thread_param_interleaver->top1 = intl_out;
                 //thread_param_interleaver->datain = scram;
                 thread_param_interleaver->datain = (unsigned char *)enc_out;
                 assert (pthread_create(&thread_interleaver, &attr_thread_interleaver, interleaver, (void *)thread_param_interleaver) == 0);
                 assert (pthread_join(thread_interleaver, NULL) == 0);
                 free(thread_param_interleaver);
              #endif
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

	      #ifndef THREAD_PER_TASK
          MOD_QPSK(OUTPUT_LEN, intl_out, sig_real, sig_img, in_ifft);
          #else
          /* Parse the structure that will be sent as arguments for the viterbi encoder fx */
          struct args_qpsk *thread_param_qpsk = (struct args_qpsk *)malloc(sizeof(struct args_qpsk));
          thread_param_qpsk->bitlen = OUTPUT_LEN;
          thread_param_qpsk->bitstream = intl_out;
          thread_param_qpsk->QPSK_real = sig_real;
          thread_param_qpsk->QPSK_img = sig_img;
          thread_param_qpsk->obuf = in_ifft;
          assert (pthread_create(&thread_qpsk_mod, &attr_thread_qpsk_mod, MOD_QPSK, (void *)thread_param_qpsk) == 0);
          assert (pthread_join(thread_qpsk_mod, NULL) == 0);
          free(thread_param_qpsk);
          #endif

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

	      #ifndef THREAD_PER_TASK
          pilotInsertion(ifft_in, pilot_out);
          #else
          /* Parse the structure that will be sent as arguments for the pilot nsertion fx */
          struct args_pilot *thread_param_pilot = (struct args_pilot *)malloc(sizeof(struct args_pilot));
          thread_param_pilot->idata = ifft_in;
          thread_param_pilot->odata = pilot_out;
          assert (pthread_create(&thread_pilot, &attr_thread_pilot, pilotInsertion, (void *)thread_param_pilot) == 0);
          assert (pthread_join(thread_pilot, NULL) == 0);
          free(thread_param_pilot);
          #endif

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
          float random_wait1;
          float random_wait2;
          float run_on_cpu;
          
          fft_id = 1;

          #ifdef FFT_HW
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
                          hw_fft_busy = 0;
                          random_wait = rand() / (float) RAND_MAX;
                          if (random_wait < FFT_WAIT_PROBABILITY) {
                              while (1) {
                                  random_wait_time_in_us = rand() % 20;
                                  random_wait_time(random_wait_time_in_us);
                                  random_wait1 = rand() / (float) RAND_MAX;
                                  if (random_wait1 > FFT_WAIT_PROBABILITY) {
                                      fft_id = 1;
                                      break;
                                  }
                                  random_wait2 = rand() / (float) RAND_MAX;
                                  if (random_wait2 > FFT_WAIT_PROBABILITY * FFT_WAIT_PROBABILITY) {
                                      fft_id = 2;
                                      break;
                                  }
                              }
                          }
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
                  #else 
                      run_on_cpu = rand() / (float) RAND_MAX;
                      if (run_on_cpu > FFT_RUN_ON_CPU_PROBABILITY) {
                          random_wait = rand() / (float) RAND_MAX;
                          while (random_wait < FFT_WAIT_PROBABILITY) {
                              random_wait_time_in_us = rand() % 20;
                              random_wait_time(random_wait_time_in_us);
                              random_wait = rand() / (float) RAND_MAX;
                          }
                          hw_fft_busy = 0;
                      } else {
                          hw_fft_busy = 1;
                      }
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

          #ifdef FFT_HW 
	      if (sem_wait(&mutex) != 0) {
                  printf("[ERROR] Semaphore wait failed ...\n");
                  exit(-1);
              }
	      #endif

	      #ifndef THREAD_PER_TASK
          ifft_hs(fft_id, pilot_out, FFT_N, hw_fft_busy);
          #else
          /* Parse the structure that will be sent as arguments for the ifft_hs fx */
          struct args_ifft *thread_param_ifft = (struct args_ifft *)malloc(sizeof(struct args_ifft));
          thread_param_ifft->fft_id = fft_id;
          thread_param_ifft->fdata = pilot_out;
          thread_param_ifft->n = FFT_N;
          thread_param_ifft->hw_fft_busy = hw_fft_busy;
          assert (pthread_create(&thread_ifft, &attr_thread_ifft, ifft_hs, (void *)thread_param_ifft) == 0);
          assert (pthread_join(thread_ifft, NULL) == 0);
          free(thread_param_ifft);
          #endif

          #ifdef FFT_HW 
              if (sem_post(&mutex) != 0) {
                  printf("[ERROR] Semaphore post failed ...\n");
                  exit(-1);
              }
	      #endif

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

          #ifndef THREAD_PER_TASK
          cyclicPrefix(pilot_out, &cyclic_out[i*(TOTAL_LEN)], FFT_N, CYC_LEN);
          #else
          /* Parse the structure that will be sent as arguments for the ifft_hs fx */
          struct args_cyclic_prefix *thread_param_crc = (struct args_cyclic_prefix *)malloc(sizeof(struct args_cyclic_prefix));
          thread_param_crc->iData = pilot_out;
          thread_param_crc->oData = &cyclic_out[i*(TOTAL_LEN)];
          thread_param_crc->len = FFT_N;
          thread_param_crc->preLen = CYC_LEN;
          assert (pthread_create(&thread_crc, &attr_thread_crc, cyclicPrefix, (void *)thread_param_crc) == 0);
          assert (pthread_join(thread_crc, NULL) == 0);
          free(thread_param_crc);
          #endif

          #ifndef PRINT_BLOCK_EXECUTION_TIMES
          #ifndef PAPI
          clock_gettime(CLOCK_MONOTONIC, &end1);
          exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
          printf("[INFO] TX-chain execution time (ns): %f\n", exec_time);
          #endif
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
      //printf("- RF file dump!!\n");
      //rfInfWrite(txdata, TX_DATA_LEN);
      #ifndef PAPI
      for(i = 0; i < TX_DATA_LEN; i++){
      	fprintf(txdata_file,"%f %f\n", txdata[i].real, txdata[i].imag);
      }
      #endif
      
      //printf("- completion!!\n");
      #endif

      frame_count++;
   }

   fclose(txdata_file);

   #ifdef FFT_HW
       printf("[INFO] Count of IFFT on A53: %d\n", fft_a53_count);
       printf("[INFO] Count of IFFT on ACC: %d\n\n\n", fft_acc_count);
   #endif

}
