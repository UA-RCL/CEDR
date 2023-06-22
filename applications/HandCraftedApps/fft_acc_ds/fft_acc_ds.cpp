//#include <iostream>
//#include <cstdio>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
//#include <cstring>
#include <fcntl.h>
//#include <sys/types.h>
#include "debug.hpp"
#include <time.h>
#include <sys/mman.h>

// Libraries for FFT on CPU
//#include <fftw3.h>
//#include <complex.h>

// RUNTIME PARAMETERS
#define SEC2NANOSEC 1000000000

// FFT PARAMETERS
#define DIM 128
#define TYPE float

// ACCELERATOR (FFT, DMA) ADDRESSES & OFFSETS
#define DMA_CONTROL_BASE_ADDR   0xA0000000
#define FFT_CONTROL_BASE_ADDR   0xA0030000

// Control Registers of DMA Memory Mapped to Stream Interface
#define DMA_OFFSET_MM2S_CONTROL  0  //00h
#define DMA_OFFSET_MM2S_STATUS   1  //04h
#define DMA_OFFSET_MM2S_SRCLWR   6  //18h
#define DMA_OFFSET_MM2S_SRCUPR   7  //1Ch // Only used when address space > 32 bits
#define DMA_OFFSET_MM2S_LENGTH   10 //28h
// Control Registers of DMA Stream to Memory Mapped Interface
#define DMA_OFFSET_S2MM_CONTROL  12 //30h
#define DMA_OFFSET_S2MM_STATUS   13 //34h
#define DMA_OFFSET_S2MM_SRCLWR   18 //48h
#define DMA_OFFSET_S2MM_SRCUPR   19 //4Ch
#define DMA_OFFSET_S2MM_LENGTH   22 //58h

// DECLARING GLOBAL VARIABLES
  // CPU
TYPE *fft_read_input, *_fft_read_input; // MOD2
TYPE *fft_input_ddr, *fft_output_ddr, *_fft_input_ddr, *_fft_output_ddr;
TYPE *fft_output_ref, *_fft_output_ref; // MOD2
FILE *fft_input_file, *_fft_input_file, *fft_refout_file, *_fft_refout_file;

  // ACC
static char attr[1024];
unsigned int *fft_control_base_addr;
TYPE  *udmabuf_base_addr;
TYPE  *fft_input_udmabuf, *fft_output_udmabuf;

unsigned long udmabuf_phys_addr;
int fd_udmabuf;

int fft_control_fd;
int dma_control_fd;
unsigned int *dma_control_base_addr;

#include "dma.h"
#include "fft.h"

// CONSTRUCTOR
void __attribute__((constructor)) setup(void) {
  // DEBUG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if defined(DEBUG)
  static char FUNC_NAME[100] = "constructor";
  static char FFT_READ_INPUT[100] = "fft_read_input";
  static char _FFT_READ_INPUT[100] = "_fft_read_input";
  static char FFT_OUTPUT_REF[100] = "fft_output_ref";
  static char _FFT_OUTPUT_REF[100] = "_fft_output_ref";
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#endif

  printf("[fft_acc] Running Constructor!\n");
  // ACCELERATOR POINTERS
  init_dma();
  printf("[fft_acc] DMA initialized!\n");
  udmabuf_base_addr = init_udmabuf(fd_udmabuf);
  printf("[fft_acc] udmabuf base address received!\n");

  init_fft();
  printf("[fft_acc] Finished initialization of FFT IP!\n");
  
  // DATA VARIABLES
  fft_input_udmabuf = &udmabuf_base_addr[0];
  fft_output_udmabuf = &udmabuf_base_addr[DIM * 2];

  fft_read_input = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));
  _fft_read_input = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));

  fft_input_ddr = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));
  _fft_input_ddr = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));
  fft_output_ddr = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));
  _fft_output_ddr = (TYPE *) malloc(DIM * 2 * sizeof(TYPE));

  //fft_input_ref = malloc(DIM * 2 * sizeof(TYPE));
  fft_output_ref = (TYPE *) calloc(DIM * 2, sizeof(TYPE));
  _fft_output_ref = (TYPE *) calloc(DIM * 2, sizeof(TYPE));

  // TEMPORARY: Read files in constructor
  fft_input_file = fopen("input/fft_input.txt", "r");
  for(size_t i = 0; i < (DIM * 2); i++) {
    fscanf(fft_input_file, "%f", &(fft_read_input[i]));
  }
#if defined(DEBUG)
  //DEBUG
  PRINT_ARRAY(fft_read_input, (DIM*2), FFT_READ_INPUT, FUNC_NAME, 0);
#endif

  for(size_t i = 0; i < (DIM * 2); i++) {
    fscanf(fft_input_file, "%f", &(_fft_read_input[i]));
  }
  fclose(fft_input_file);
#if defined(DEBUG)
  //DEBUG
  PRINT_ARRAY(_fft_read_input, (DIM*2), _FFT_READ_INPUT, FUNC_NAME, 0);
#endif

  fft_refout_file = fopen("input/fft_refout.txt", "r");
  for(size_t i=0; i < (DIM * 2); i++) {
    fscanf(fft_refout_file, "%f", &(fft_output_ref[i]));
  }

#if defined(DEBUG)
  //DEBUG
  PRINT_ARRAY(fft_output_ref, (DIM*2), FFT_OUTPUT_REF, FUNC_NAME, 0);
#endif

  for(size_t i=0; i < (DIM * 2); i++) {
    fscanf(fft_refout_file, "%f", &(_fft_output_ref[i]));
  }
  fclose(fft_refout_file);

#if defined(DEBUG)
  //DEBUG
  PRINT_ARRAY(_fft_output_ref, (DIM*2), _FFT_OUTPUT_REF, FUNC_NAME, 0);
#endif
  printf("[fft_acc] End of Constructor!\n");
}



// DESTRUCTOR
void __attribute__ ((destructor)) clean_app(void) {
  printf("[fft_acc] Running Destructor!\n");
  free(fft_input_ddr);
  free(_fft_input_ddr);
  free(fft_output_ddr);
  free(_fft_output_ddr);
  //free(fft_input_ref);
  free(fft_read_input);
  free(_fft_read_input);
  free(fft_output_ref);
  free(_fft_output_ref);

  munmap(fft_control_base_addr, getpagesize());
  close(fft_control_fd);
  close_dma();
  printf("[fft_acc] Finished Destructor!\n");
}


//#########################################################################
// Function 1 Read fft input from file
//#########################################################################
extern "C" void FFT_head_node(void) {
  static bool BUF_SEL = true;
  struct timespec sleepTime;
  sleepTime.tv_sec = 0;
  sleepTime.tv_nsec = 300000; // 0.3 millisecond

#if defined(DEBUG)
  static int ITER = 0;
  static char FUNC_NAME[100] = "FFT_head_node";
  static char FFT_INPUT_DDR[100] = "fft_input_ddr";
  static char _FFT_INPUT_DDR[100] = "_fft_input_ddr";
  static char FFT_READ_INPUT[100] = "fft_read_input";
  static char _FFT_READ_INPUT[100] = "_fft_read_input";
#endif

  if (BUF_SEL) {
    // copy read data
    for (size_t i = 0; i < (DIM * 2); i++) {
      fft_input_ddr[i] = fft_read_input[i];
    }
  }
  else {
    for (size_t i = 0; i < (DIM * 2); i++) {
      _fft_input_ddr[i] = _fft_read_input[i];
    }
  }

#if defined(DEBUG)
  //DEBUG
  printf("[FFT_head_node %d] ############################### FFT_head_node - Iteration %d #############################\n", ITER, ITER);
  printf("[FFT_head_node %d] Value of ITER iteration is %d and BUF_SEL value is %s\n", ITER, ITER, BUF_SEL?"true":"false");

  PRINT_ARRAY(fft_read_input, (DIM*2), FFT_READ_INPUT, FUNC_NAME,  ITER);

  PRINT_ARRAY(fft_input_ddr, (DIM*2), FFT_INPUT_DDR, FUNC_NAME, ITER);

  PRINT_ARRAY(_fft_read_input, (DIM*2), _FFT_READ_INPUT, FUNC_NAME, ITER);

  PRINT_ARRAY(_fft_input_ddr, (DIM*2), _FFT_INPUT_DDR, FUNC_NAME, ITER);
  printf("[FFT_head_node %d] ######################################################################################\n", ITER);

  ITER ++;
#endif

  BUF_SEL = ! BUF_SEL;
  nanosleep(&sleepTime, NULL);
}



//#########################################################################
// Function 2 Setup FFT accelerator to perform FFT operation
//#########################################################################
extern "C" void FFT_accel(void){
  static bool BUF_SEL = true;
  struct timespec sleepTime;
  sleepTime.tv_sec = 0;
  sleepTime.tv_nsec = 100000; // 0.1 millisecond

#if defined(DEBUG)
  static int ITER = 0;
  static char FUNC_NAME[100] = "FFT_accel";
  static char FFT_INPUT_DDR[100] = "fft_input_ddr";
  static char _FFT_INPUT_DDR[100] = "_fft_input_ddr";
  static char FFT_INPUT_UDMABUF[100] = "fft_input_udmabuf";
  static char FFT_OUTPUT_DDR[100] = "fft_output_ddr";
  static char _FFT_OUTPUT_DDR[100] = "_fft_output_ddr";
  static char FFT_OUTPUT_UDMABUF[100] = "fft_output_udmabuf";
#endif

  // Copy data from DDR to udmabuf
  if (BUF_SEL) {
    for (size_t i = 0; i < (DIM*2); i++) {
      fft_input_udmabuf[i] = fft_input_ddr[i];
    }
  }
  else {
    for (size_t i = 0; i < (DIM * 2); i++) {
      fft_input_udmabuf[i] = _fft_input_ddr[i];
    }
  }


  // configure fft accelerator
  config_fft(fft_control_base_addr, log2(DIM));
  printf("[FFT_DS] Finished configuring FFT IP\n");

  // Reset DMA
  reset_dma();
  printf("[FFT_DS] Finished resetting DMA!\n");

  // setup receiver
  setup_rx();
  
  // setup transmitter (send data to fft accel. over udmabuf)
  setup_tx();

  // wait for TX to complete
  dma_wait_for_tx_complete();
  printf("[fft_acc] TX setup complete!\n");

  // wait for accelerator (RX) to complete
  dma_wait_for_rx_complete();
  printf("[fft_acc] FFT accelerator done processing data!\n");
  
  // receive data from udmabuf
  if (BUF_SEL) {
    for (size_t i = 0; i < (DIM * 2); i++) {
      fft_output_ddr[i] = fft_output_udmabuf[i];
    }
  }
  else {
    for (size_t i = 0; i < (DIM * 2); i++) {
      _fft_output_ddr[i] = fft_output_udmabuf[i];
    }
  }

#if defined(DEBUG)
  //DEBUG

  printf("[FFT_accel %d] ############################### FFT_accel - Iteration %d #############################\n", ITER, ITER);
  printf("[FFT_accel %d] Value of ITER iteration is %d and BUF_SEL value is %s\n", ITER, ITER, BUF_SEL?"true":"false");

  PRINT_ARRAY(fft_input_ddr, (DIM*2), FFT_INPUT_DDR, FUNC_NAME, ITER);
  PRINT_ARRAY(_fft_input_ddr, (DIM*2), _FFT_INPUT_DDR, FUNC_NAME, ITER);
  PRINT_ARRAY(fft_input_udmabuf, (DIM*2), FFT_INPUT_UDMABUF, FUNC_NAME, ITER);

  printf("---------------------- Running FFT Accel ----------------------\n");

  PRINT_ARRAY(fft_output_udmabuf, (DIM*2), FFT_OUTPUT_UDMABUF, FUNC_NAME, ITER);
  PRINT_ARRAY(fft_output_ddr, (DIM*2), FFT_OUTPUT_DDR, FUNC_NAME, ITER);
  PRINT_ARRAY(_fft_output_ddr, (DIM*2), _FFT_OUTPUT_DDR, FUNC_NAME, ITER);

  printf("[FFT_accel %d] ######################################################################################\n", ITER);

  ITER ++;
#endif

  BUF_SEL = ! BUF_SEL;
  nanosleep(&sleepTime, NULL);

}


//#########################################################################
// Function 3 Validate FFT accelerator output with reference output (FILE)
//#########################################################################
extern "C" void FFT_output(void) {
  static bool BUF_SEL = true;
  //bool MISMATCH = false;
  struct timespec sleepTime;
  sleepTime.tv_sec = 0;
  sleepTime.tv_nsec = 400000; // 0.4 millisecond

#if defined(DEBUG)
  static int ITER = 0;
  static char FUNC_NAME[20] = "FFT_output";
  static char FFT_OUTPUT_DDR[20] = "fft_output_ddr";
  static char _FFT_OUTPUT_DDR[20] = "_fft_output_ddr";
  static char FFT_OUTPUT_REF[20] = "fft_output_ref";
  static char _FFT_OUTPUT_REF[20] = "_fft_output_ref";
#endif


  if (BUF_SEL) {
    // For loop matching each position and sending output
    // if MISMATCH == true, print the mismatch message (and index?)
    check_result(fft_output_ref, fft_output_ddr);
  }
  else {
    check_result(_fft_output_ref, _fft_output_ddr);
  }

#if defined(DEBUG)
  //DEBUG

  printf("[FFT_output %d] ############################### FFT_output - Iteration %d #############################\n", ITER, ITER);
  printf("[FFT_output %d] Value of ITER iteration is %d and BUF_SEL value is %s\n", ITER, ITER, BUF_SEL?"true":"false");

  PRINT_ARRAY(fft_output_ref, (DIM*2), FFT_OUTPUT_REF, FUNC_NAME, ITER);

  PRINT_ARRAY(fft_output_ddr, (DIM*2), FFT_OUTPUT_DDR, FUNC_NAME, ITER);

  PRINT_ARRAY(_fft_output_ref, (DIM*2), _FFT_OUTPUT_REF, FUNC_NAME, ITER);

  PRINT_ARRAY(_fft_output_ddr, (DIM*2), _FFT_OUTPUT_DDR, FUNC_NAME, ITER);

  printf("[FFT_output %d] ######################################################################################\n", ITER);

  ITER ++;
#endif

  BUF_SEL = ! BUF_SEL;
  nanosleep(&sleepTime, NULL);

}

