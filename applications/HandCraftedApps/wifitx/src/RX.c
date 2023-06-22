#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "common.h"
#include "fft_hs.h"
#include "scrambler_descrambler.h"
#include "CyclicPrefix.h"
#include "Preamble_ST_LG.h"
#include "viterbi.h"
#include "baseband_lib.h"
#include "interleaver_deintleaver.h"
#include "qpsk_Mod_Demod.h"
#include "ch_Est_Equa.h"
#include "detection.h"
#include "rfInf.h"
#include "rt_nonfinite.h"
#include "channel_Eq.h"
#include "channel_Eq_terminate.h"
#include "channel_Eq_initialize.h"
#include "decode.h"
#include "datatypeconv.h"
#include "pilot.h"
#include "equalizer.h"
#include "rfInf.h"
#include "rf_interface.h"

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "fft_hwa.h"
#include "dma.h"

#define SEC2NANOSEC 1000000000

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

static char         attr[1024];

signed char dec_in[OUTPUT_LEN];
unsigned char dec_out[USR_DAT_LEN];
signed char dec_pun_out[OUTPUT_LEN];
signed char deintl_out[OUTPUT_LEN];
double out_real[OUTPUT_LEN];
double out_img[OUTPUT_LEN];
signed char outbit[OUTPUT_LEN];
unsigned char descram[USR_DAT_LEN*SYM_NUM+1];
int hw_fft_busy;

unsigned int *base_addr;

int fft_a53_count = 0;
int fft_acc_count = 0;

int main() {

    int j;

    int decoderId;
    comp_t out_fd[FFT_N];
	comp_t pilot_rm[INPUT_LEN];
    float *offt;
    float pilotdata_rx[PILOT_LEN];
    float out_fft_c2f[INPUT_LEN * 2];
    int frameCount = 0;
	FILE *fp;
    struct timespec start1, end1;
    float exec_time;

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
        config_fft(fft1_control_base_addr);

        // Virtual Address to udmabuf Buffer
        init_udmabuf1();

    #endif

    #ifdef FFT2_HW
        // Virtual Address to DMA Control Slave
        init_dma2();

        init_fft2();
        config_fft(fft2_control_base_addr);

        // Virtual Address to udmabuf Buffer
        init_udmabuf2();

    #endif

    // create_frameDetection(OSRATE); // up sample rate=4
    #ifdef TARGET
        create_rfInf(RXMODE, RFCARD, INT_2BYTE, NO_SMPERR, 0);
    #else
        create_rfInf(RXMODE, DATFILE, INT_2BYTE, NO_SMPERR, 0);
    #endif

    init_viterbiDecoder();
    decoderId = get_viterbiDecoder();
    set_viterbiDecoder(decoderId);
    init_equalizer();

    int frame_count = 0;

    while(frame_count < NUM_FRAMES) {

       //............... Frame Detection ........................
       if(frameDetection() == 0) frameDetection();

       // clean RX buffer
       for(j=0; j<SYM_NUM*USR_DAT_LEN; j++) descram[j] = 0;

       for(j=0; j<SYM_NUM; j++) {

           #ifndef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           //###############################################################
           //## Payload Extraction
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           payloadExt(out_fd);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-PayloadExt execution time (ns): %f\n", exec_time);
           #endif
  
           //###############################################################
           //## FFT
           //###############################################################
           #ifdef FFT_HW
               srand(clock());
               hw_fft_busy = rand() % 2;
               #ifdef FFT_HW_ONLY
                   hw_fft_busy = 0;
               #endif
               if (hw_fft_busy == 1) fft_a53_count++;
               if (hw_fft_busy == 0) fft_acc_count++;
           #else
               hw_fft_busy = 1;
               #ifdef PRINT_BLOCK_EXECUTION_TIMES
               clock_gettime(CLOCK_MONOTONIC, &start1);
               #endif
           #endif

           fft_hs(2, out_fd, FFT_N, hw_fft_busy);
           
           #ifdef FFT_HW
               #ifdef PRINT_BLOCK_EXECUTION_TIMES
               clock_gettime(CLOCK_MONOTONIC, &end1);
               exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
               printf("[INFO] RX-FFT execution time (ns): %f\n", exec_time);
               #endif
           #endif

           //offt = (float *)out_fd;

           //###############################################################
           //## Pilot Extraction
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           pilotExtract(out_fd, pilotdata_rx);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-PilotExt execution time (ns): %f\n", exec_time);
           #endif
  
           //....... Channel Estimation and Equalization ............
           // if(j == 0) equalization(pilotdata_rx, offt, out_fft_c2f, 0); // estimation
           // equalization(pilotdata_rx, offt, out_fft_c2f, 1); // equalization

           //###############################################################
           //## Pilot Removal
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           pilotRemove(FFT_N, out_fd, pilot_rm); 

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-PilotRmv execution time (ns): %f\n", exec_time);
           #endif
  
           for(int i = 0; i < INPUT_LEN; i++){
               out_fft_c2f[2*i] = pilot_rm[i].real;
               out_fft_c2f[2*i + 1] = pilot_rm[i].imag;
           }

           //###############################################################
           //## QPSK Demodulation
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           DeMOD_QPSK(INPUT_LEN, pilot_rm, outbit);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-QPSK execution time (ns): %f\n", exec_time);
           #endif
  
           //###############################################################
           //## Deinterleaver
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           deinterleaver(outbit,OUTPUT_LEN,deintl_out);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-Deinterleaver execution time (ns): %f\n", exec_time);
           #endif
  
           //format conversion
           #ifdef HARDINPUT
                     formatConversion(PUNC_RATE_1_2, deintl_out, dec_in);
           #endif
                     // depuncturing
           #ifdef HARDINPUT
                     viterbi_depuncturing(PUNC_RATE_1_2, dec_in, dec_pun_out);
           #else
                     viterbi_depuncturing(PUNC_RATE_1_2, deintl_out, dec_pun_out);
           #endif

           //###############################################################
           //## Viterbi Decoder
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           viterbi_decoding(decoderId, dec_pun_out, dec_out);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-Decoder execution time (ns): %f\n", exec_time);
           #endif
  
           //###############################################################
           //## Descrambler
           //###############################################################
           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &start1);
           #endif

           descrambler(USR_DAT_LEN, dec_out, &descram[j*USR_DAT_LEN]);

           #ifdef PRINT_BLOCK_EXECUTION_TIMES
           clock_gettime(CLOCK_MONOTONIC, &end1);
           exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
           printf("[INFO] RX-Decrambler execution time (ns): %f\n\n", exec_time);
           #endif
  
       #ifndef PRINT_BLOCK_EXECUTION_TIMES
       clock_gettime(CLOCK_MONOTONIC, &end1);
       exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
       printf("[INFO] RX-chain execution time (ns): %f\n", exec_time);
       #endif

       }

       //###############################################################
       //## Message Decoder
       //###############################################################
       #ifdef PRINT_BLOCK_EXECUTION_TIMES
       clock_gettime(CLOCK_MONOTONIC, &start1);
       #endif

       messagedecoder((unsigned char *)descram);

       #ifdef PRINT_BLOCK_EXECUTION_TIMES
       clock_gettime(CLOCK_MONOTONIC, &end1);
       exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
       printf("\n[INFO] RX-MsgDecode execution time (ns): %f\n", exec_time);
       #endif

       frame_count++;
  
    }
   
   #ifdef FFT_HW
       printf("\n[INFO] Count of FFT on A53: %d\n", fft_a53_count);
       printf("[INFO] Count of FFT on ACC: %d\n\n\n", fft_acc_count);
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
