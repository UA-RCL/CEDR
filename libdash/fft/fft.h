#pragma once

#include "dash_types.h"
#include "platform.h"
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#if DASH_PLATFORM == DASH_JETSONAGX
  #error "Current platform does not support the FFT module. You likely want the FFT implementation of the GPU module instead"
#endif

// What is the baseline data type that the accelerator works with? 
// We need to convert whatever baseline type we're given into this
typedef float fft_re_type; 

// And then we pack that data type together into re/im complex inputs
typedef struct fft_cmplx_type {
  fft_re_type re;
  fft_re_type im;
} fft_cmplx_type;

// Size of array in bytes divided by size of first element in bytes == number of elements in array
#define NUM_FFTS (sizeof(FFT_CONTROL_BASE_ADDRS) / sizeof(FFT_CONTROL_BASE_ADDRS[0]))

#define UDMABUF_PARTITION_SIZE (FFT_UDMABUF_SIZE / NUM_FFTS)
// Our FFT IP is configured for up to 2048-Pt, each element is of size "sizeof(fft_cmplx_type)" and we need 2x that for input + output
#define REQUIRED_BUFFER_SIZE_FLT (2 * 2048 * sizeof(fft_cmplx_type))
static_assert(UDMABUF_PARTITION_SIZE >= REQUIRED_BUFFER_SIZE_FLT, "Current udmabuf size is too small to support this many FFT accelerators!");

//#define __DASH_FFT_DEBUG__

#ifdef LOG
#undef LOG
#endif

#ifdef __DASH_FFT_DEBUG__
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
#endif

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
volatile unsigned int* init_fft(unsigned int FFT_CONTROL_BASE_ADDR) {
  int fd;
  volatile unsigned int* virtual_addr;

  if (FFT_CONTROL_BASE_ADDR == 0x00000000) {
    LOG("[fft] Trying to initialize FFT, but its base address is 0, I don't think one is available!\n");
    exit(1);
  }

  LOG("[fft] Initializing FFT at control address 0x%x\n", FFT_CONTROL_BASE_ADDR);
  // Open device memory in order to get access to DMA control slave
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[fft] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  // Obtain virtual address to DMA control slave through mmap
  virtual_addr = (volatile unsigned int *)mmap(nullptr,
                                            getpagesize(),
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED,
                                            fd,
                                            FFT_CONTROL_BASE_ADDR);

  if (virtual_addr == MAP_FAILED) {
    // TODO: does mmap set errno? might be nice to perror here
    LOG("[fft] Can't obtain memory map to FFT control slave. Exiting ...\n");
    exit(1);
  }

  close(fd);
  return virtual_addr;
}


//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void inline fft_write_reg(volatile unsigned int *base, unsigned int offset, unsigned int data) { 
  *(base + offset) = data; 
}

//###################################################################################
// Function that configures FFT accelerator for an inverse transform
//###################################################################################
void config_ifft(volatile unsigned int *base, unsigned int size) {
#if defined(FFT_CONFIG_VIA_GPIO)
  // So this functionality is configuring the Xilinx FFT through an AXI GPIO IP
  // GPIO 1 => data at 0x0000, tri at 0x0004, connected to the "tvalid" pin of FFT IP
  // GPIO 2 => data at 0x0008, tri at 0x000C, connected to the "tdata" pin of FFT IP
  
  // First, ensure that both GPIOs are set as write-mode
  fft_write_reg(base, 0x1, 0x0);
  fft_write_reg(base, 0x3, 0x0);
  // Then, write the configuration data (is_forward | log2(size))
  fft_write_reg(base, 0x2, (0x0 << 8 | size));
  // Write that the configuration data is valid
  fft_write_reg(base, 0x0, 0x1);
  // Delay a couple cycles
  volatile uint32_t dummy_var = 0;
  while (dummy_var < FFT_GPIO_CONFIG_DELAY) { dummy_var++; }
  // Release the configuration tdata valid bit
  fft_write_reg(base, 0x0, 0x0);
#else
  // Config via fft_axi_config
  fft_write_reg(base, 0x0, (0x0 << 8 | size));
#endif
}

//###################################################################################
// Function that configures FFT accelerator for a forward transform
//###################################################################################
void config_fft(volatile unsigned int *base, unsigned int size) {
#if defined(FFT_CONFIG_VIA_GPIO)
  // So this functionality is configuring the Xilinx FFT through an AXI GPIO IP
  // GPIO 1 => data at 0x0000, tri at 0x0004, connected to the "tvalid" pin of FFT IP
  // GPIO 2 => data at 0x0008, tri at 0x000C, connected to the "tdata" pin of FFT IP

  // First, ensure that both GPIOs are set as write-mode
  fft_write_reg(base, 0x1, 0x0);
  fft_write_reg(base, 0x3, 0x0);
  // Then, write the configuration data (is_forward | log2(size))
  fft_write_reg(base, 0x2, (0x1 << 8 | size));
  // Write that the configuration data is valid
  fft_write_reg(base, 0x0, 0x1);
  // Delay a couple cycles
  volatile uint32_t dummy_var = 0;
  while (dummy_var < FFT_GPIO_CONFIG_DELAY) { dummy_var++; }
  // Release the configuration tdata valid bit
  fft_write_reg(base, 0x0, 0x0);
#else
  // Config via fft_axi_config
  fft_write_reg(base, 0x0, (0x1 << 8 | size));
#endif
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void inline close_fft(volatile unsigned int* virtual_addr) {
  munmap((unsigned int*)virtual_addr, getpagesize());
}

#if defined(FFT_GPIO_RESET_BASE_ADDRS)
volatile unsigned int* init_fft_reset(unsigned int FFT_GPIO_RESET_BASE_ADDR) {
  int fd;
  volatile unsigned int* virtual_addr;

  if (FFT_GPIO_RESET_BASE_ADDR == 0x00000000) {
    LOG("[fft] Trying to initialize FFT, but its base address is 0, I don't think one is available!\n");
    exit(1);
  }

  LOG("[fft] Initializing FFT GPIO reset at control address 0x%x\n", FFT_GPIO_RESET_BASE_ADDR);
  // Open device memory in order to get access to DMA control slave
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[fft] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  // Obtain virtual address to DMA control slave through mmap
  virtual_addr = (volatile unsigned int *)mmap(nullptr,
                                            getpagesize(),
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED,
                                            fd,
                                            FFT_GPIO_RESET_BASE_ADDR);

  if (virtual_addr == MAP_FAILED) {
    // TODO: does mmap set errno? might be nice to perror here
    LOG("[fft] Can't obtain memory map to FFT control slave. Exiting ...\n");
    exit(1);
  }

  close(fd);
  return virtual_addr;
}

void reset_fft_and_dma(volatile unsigned int *base) {
  // It's a single output GPIO, so make sure that it's set to output
  // And then as it's active low, hold it low for a bit and then release (set back to 1)
  // GPIO 1 => data at 0x0000, tri at 0x0004

  // Note: technically we're writing to a different reg than is typically used with fft_write_reg
  // But the functionality we need is exactly the same, so it's fine.
  // Set the GPIO as write mode
  fft_write_reg(base, 0x1, 0x0);
  // Enable the reset
  fft_write_reg(base, 0x0, 0x0);
  // Delay a bit
  volatile uint32_t dummy_var = 0;
  while (dummy_var < 10) { dummy_var++; }
  // Release the reset
  fft_write_reg(base, 0x0, 0x1);
}

void inline close_fft_reset(volatile unsigned int *virtual_addr) {
  munmap((unsigned int*) virtual_addr, getpagesize());
}
#endif
