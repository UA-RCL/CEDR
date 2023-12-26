#pragma once

#include "dash_types.h"
#include "platform.h"
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#if DASH_PLATFORM == DASH_JETSONAGX
  #error "Current platform does not support the ZIP module. You likely want the ZIP implementation of the GPU module instead"
#endif
#if DASH_PLATFORM == DASH_ZCU102_2020_2
  #error "Current platform does not support the ZIP module. Make sure you are using the correct platform information"
#endif

// What is the baseline data type that the accelerator works with?
// We need to convert whatever baseline type we're given into this
typedef float zip_re_type;

// And then we pack that data type together into re/im complex inputs
typedef struct zip_cmplx_type {
  zip_re_type re;
  zip_re_type im;
} zip_cmplx_type;

// Size of array in bytes divided by size of first element in bytes == number of elements in array
#define NUM_ZIPS (sizeof(ZIP_CONTROL_BASE_ADDRS) / sizeof(ZIP_CONTROL_BASE_ADDRS[0]))

#define UDMABUF_PARTITION_SIZE (ZIP_UDAMBUF_SIZE/NUM_ZIPS)

// ZIP is configured to support up to 4096 input size
//Make sure that our udmabuf partitions are sized such that we can hold at least one INPUT and the OUTPUT for each ZIP in non-conflicting buffers
#define REQUIRED_BUFFER_SIZE ((4096*3) * sizeof(float))
static_assert(UDMABUF_PARTITION_SIZE >= REQUIRED_BUFFER_SIZE, "Current udmabuf size is too small to support this many ZIPs!");

//#define __DASH_ZIP_DEBUG__

#ifdef LOG
#undef LOG
#endif

#ifdef __DASH_ZIP_DEBUG__
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
#endif

//###################################################################################
// Function that user code calls to actually perform a full ZIP on the accelerator
//###################################################################################
void zip_accel(zip_re_type* input0, zip_re_type* input1, zip_re_type* output, int size, int op);

//###################################################################################
// Function to initialize memory maps to ZIP
//###################################################################################
volatile unsigned int* init_zip(unsigned int ZIP_CONTROL_BASE_ADDR) {
  int fd;
  volatile unsigned int* virtual_addr;

  if (ZIP_CONTROL_BASE_ADDR == 0x00000000) {
    LOG("[zip] Trying to initialize ZIP, but its base address is 0, I don't think one is available!\n");
    exit(1);
  }

  //printf("[zip] Initializing ZIP at control address 0x%x\n", ZIP_CONTROL_BASE_ADDR);
  // Open device memory in order to get access to DMA control slave
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[zip] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  // Obtain virtual address to DMA control slave through mmap
  virtual_addr = (volatile unsigned int *)mmap(nullptr,
                                            getpagesize(),
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED,
                                            fd,
                                            ZIP_CONTROL_BASE_ADDR);

  if (virtual_addr == MAP_FAILED) {
    // TODO: does mmap set errno? might be nice to perror here
    LOG("[zip] Can't obtain memory map to ZIP control slave. Exiting ...\n");
    exit(1);
  }

  close(fd);
  return virtual_addr;
}


//###################################################################################
// Function to Write Data to ZIP Control Register
//###################################################################################
void inline zip_write_reg(volatile unsigned int *base, unsigned int offset, unsigned int data) { 
  *((unsigned int*)((char*)base + offset)) = data;
}

//###################################################################################
// Function that configures ZIP accelerator for an inverse transform
//###################################################################################
void config_zip_op(volatile unsigned int *base, unsigned int op){
  if(op==0){
    LOG("[zip] Configuring ZIP as ADD\n");
    zip_write_reg(base, ZIP_OP_OFFSET, 0x00);
  }
  else if (op==1){
    LOG("[zip] Configuring ZIP as SUB\n");
    zip_write_reg(base, ZIP_OP_OFFSET, 0x01);
  }
  else if (op==2){
    LOG("[zip] Configuring ZIP as MULT\n");
    zip_write_reg(base, ZIP_OP_OFFSET, 0x02);
  }
  else if (op==3){
    LOG("[zip] Configuring ZIP as DIV\n");
    zip_write_reg(base, ZIP_OP_OFFSET, 0x03);
  }
  else if (op==4){
    LOG("[zip] Configuring ZIP as COMP_MULT\n");
    zip_write_reg(base, ZIP_OP_OFFSET, 0x04);
  }
  else{
    zip_write_reg(base, ZIP_OP_OFFSET, op);
  }
}

void config_zip_size(volatile unsigned int *base, unsigned int size){
  zip_write_reg(base, ZIP_SIZE_OFFSET, size);
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void inline close_zip(volatile unsigned int* virtual_addr) {
  munmap((unsigned int*)virtual_addr, getpagesize());
}

