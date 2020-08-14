/*
 * fft_dma.c
 *
 *  Created on: Jun 25, 2020
 *      Author: hanguang yu
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include "fft_dma.h"

static char attr[1024];

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(fft_dma_cfg *fft_dma, unsigned int offset, int data) {

  (fft_dma->dma_control_base_addr)[offset] = data;
  //   dma_control_base_addr[offset] = data;
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(fft_dma_cfg *fft_dma) {

  while (((fft_dma->dma_control_base_addr)[DMA_OFFSET_MM2S_STATUS] & 0x01) != 0x01)
    ;
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(fft_dma_cfg *fft_dma) {

  while (((fft_dma->dma_control_base_addr)[DMA_OFFSET_S2MM_STATUS] & 0x01) != 0x01)
    ;
}

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(fft_dma_cfg *fft_dma) {

  while (((fft_dma->dma_control_base_addr)[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02)
    ;
}
//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(fft_dma_cfg *fft_dma) {

  while (((fft_dma->dma_control_base_addr)[DMA_OFFSET_S2MM_STATUS] & 0x03) != 0x02)
    ;
}

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
void init_dma(fft_dma_cfg *fft_dma) {

  unsigned int addr_tmp = 0;

  // Open device memory in order to get access to DMA control slave
  fft_dma->dma_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fft_dma->dma_control_fd < 0) {
    //printf("[fft][ERROR] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /dev/mem ...\n");

  switch (fft_dma->list) {
  case (fft_dma_config::fft_0): {
    addr_tmp = DMA0_CONTROL_BASE_ADDR;
    break;
  }
  case (fft_dma_config::fft_1): {
    addr_tmp = DMA1_CONTROL_BASE_ADDR;
    break;
  }
  default: {
    break;
  }
  }

  // Obtain virtual address to DMA control slave through mmap
  fft_dma->dma_control_base_addr =
      (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fft_dma->dma_control_fd, addr_tmp);

  if (fft_dma->dma_control_base_addr == MAP_FAILED) {
    //printf("[fft][ERROR] Can't obtain memory map to DMA_%d control slave. Exiting ...\n", fft_dma->list);
    exit(1);
  }

  //printf("[fft][INFO] Successfully obtained virtual address to DMA_%d control slave ...\n", fft_dma->list);
}

void init_fft(fft_dma_cfg *fft_dma) {

  unsigned int addr_tmp = 0;

  // Open device memory in order to get access to DMA control slave
  fft_dma->fft_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fft_dma->fft_control_fd < 0) {
    //printf("[fft][ERROR] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /dev/mem ...\n");

  switch (fft_dma->list) {
  case (fft_dma_config::fft_0): {
    addr_tmp = FFT0_CONTROL_BASE_ADDR;
    break;
  }
  case (fft_dma_config::fft_1): {
    addr_tmp = FFT1_CONTROL_BASE_ADDR;
    break;
  }
  default: {
    break;
  }
  }

  // Obtain virtual address to DMA control slave through mmap
  fft_dma->fft_control_base_addr =
      (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fft_dma->fft_control_fd, addr_tmp);

  if (fft_dma->fft_control_base_addr == MAP_FAILED) {
    //printf("[fft][ERROR] Can't obtain memory map to FFT_%d control slave. Exiting ...\n", fft_dma->list);
    exit(1);
  }

  //printf("[fft][INFO] Successfully obtained virtual address to FFT_%d control slave ...\n", fft_dma->list);
}

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf_0(fft_dma_cfg *fft_dma) {

  fft_dma->fd_udmabuf = open("/dev/udmabuf0", O_RDWR | O_SYNC);
  if (fft_dma->fd_udmabuf < 0) {
    //printf("[fft][ERROR] Can't open /dev/udmabuf0. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /dev/udmabuf0 ...\n");

  //    uint32_t *base_addr = (uint32_t *)mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_SHARED, fft_dma_0.fd_udmabuf,
  //    0);
  fft_dma->udmabuf_base_addr =
      (unsigned int *)mmap(NULL, udma_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fft_dma->fd_udmabuf, 0);

  if (fft_dma->udmabuf_base_addr == MAP_FAILED) {
    //printf("[fft][ERROR] Can't obtain memory map to udmabuf buffer. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully obtained virtual address to udmabuf buffer ...\n");

  int fd_udmabuf_addr = open("/sys/class/u-dma-buf/udmabuf0/phys_addr", O_RDONLY);
  if (fd_udmabuf_addr < 0) {
    //printf("[fft][ERROR] Can't open /sys/class/u-dma-buf/udmabuf0/phys_addr. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /sys/class/u-dma-buf/udmabuf0/phys_addr ...\n");
  read(fd_udmabuf_addr, attr, 1024);
  sscanf(attr, "%llx", &(fft_dma->udmabuf_phys_addr));
  close(fd_udmabuf_addr);

  //    printf("phy address:%llx\n",fft_dma->udmabuf_phys_addr);

  // Reset DMA
  dma_write_reg(fft_dma, DMA_OFFSET_MM2S_CONTROL, 0x4);
  dma_write_reg(fft_dma, DMA_OFFSET_S2MM_CONTROL, 0x4);
  dma_wait_for_tx_idle(fft_dma);
  dma_wait_for_rx_idle(fft_dma);

  //    return base_addr;
}

void init_udmabuf_1(fft_dma_cfg *fft_dma) {

  fft_dma->fd_udmabuf = open("/dev/udmabuf1", O_RDWR | O_SYNC);
  if (fft_dma->fd_udmabuf < 0) {
    //printf("[fft][ERROR] Can't open /dev/udmabuf1. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /dev/udmabuf1 ...\n");

  fft_dma->udmabuf_base_addr =
      (unsigned int *)mmap(NULL, udma_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fft_dma->fd_udmabuf, 0);

  if (fft_dma->udmabuf_base_addr == MAP_FAILED) {
    //printf("[fft][ERROR] Can't obtain memory map to udmabuf buffer. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully obtained virtual address to udmabuf buffer ...\n");

  int fd_udmabuf_addr = open("/sys/class/u-dma-buf/udmabuf1/phys_addr", O_RDONLY);
  if (fd_udmabuf_addr < 0) {
    //printf("[fft][ERROR] Can't open /sys/class/u-dma-buf/udmabuf1/phys_addr. Exiting ...\n");
    exit(1);
  }

  //printf("[fft][INFO] Successfully opened /sys/class/u-dma-buf/udmabuf1/phys_addr ...\n");
  read(fd_udmabuf_addr, attr, 1024);
  sscanf(attr, "%llx", &(fft_dma->udmabuf_phys_addr));
  close(fd_udmabuf_addr);

  //    printf("phy address:%llx\n",fft_dma->udmabuf_phys_addr);

  // Reset DMA
  dma_write_reg(fft_dma, DMA_OFFSET_MM2S_CONTROL, 0x4);
  dma_write_reg(fft_dma, DMA_OFFSET_S2MM_CONTROL, 0x4);
  dma_wait_for_tx_idle(fft_dma);
  dma_wait_for_rx_idle(fft_dma);

  //    return base_addr;
}

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_fft_dma_tx(fft_dma_cfg *fft_dma) {

  dma_write_reg(fft_dma, DMA_OFFSET_MM2S_SRCLWR, fft_dma->udmabuf_phys_addr);
  dma_write_reg(fft_dma, DMA_OFFSET_MM2S_CONTROL, 0x3);
  dma_write_reg(fft_dma, DMA_OFFSET_MM2S_LENGTH, fft_dma->fft_dim * 2 * sizeof(TYPE));
  // printf("[ INFO] Sent data to HW accelerator over DMA ...\n");
}

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_fft_dma_rx(fft_dma_cfg *fft_dma) {

  dma_write_reg(fft_dma, DMA_OFFSET_S2MM_SRCLWR, (fft_dma->udmabuf_phys_addr + fft_dma->fft_dim * 2 * sizeof(TYPE)));
  dma_write_reg(fft_dma, DMA_OFFSET_S2MM_CONTROL, 0x3);
  dma_write_reg(fft_dma, DMA_OFFSET_S2MM_LENGTH, fft_dma->fft_dim * 2 * sizeof(TYPE));
  // printf("[ INFO] Initiated DMA receiver ...\n");
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma(fft_dma_cfg *fft_dma) {
  munmap((void*)fft_dma->dma_control_base_addr, getpagesize());
  close(fft_dma->dma_control_fd);
  // printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
  // printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}

//###################################################################################
// Function to remove memory maps to udma buffer
//###################################################################################
void close_udma_buffer(fft_dma_cfg *fft_dma) {
  munmap((void*)fft_dma->udmabuf_base_addr, udma_buffer_size);
  close(fft_dma->fd_udmabuf);
  // printf("[ INFO] Un-map of virtual address obtained to udmabuf control slave ...\n");
  // printf("[ INFO] Closing file descriptor to udmabuf control slave...\n");
}

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void fft_write_reg(fft_dma_cfg *fft_dma, unsigned int offset, unsigned int data) {

  *(fft_dma->fft_control_base_addr + offset) = data;
}

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_ifft(fft_dma_cfg *fft_dma, unsigned int size) {

  fft_write_reg(fft_dma, 0x0, size);
  // printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_fft(fft_dma_cfg *fft_dma, unsigned int size) {

  // must refer to xilinx fft IP document for details
  fft_write_reg(fft_dma, 0x0, size | 0x000100);

  // printf("[ INFO] Configured FFT IP ...\n");
}
//###################################################################################
// Function to remove memory maps to fft controller
//###################################################################################
void close_fft(fft_dma_cfg *fft_dma) {
  munmap((void*)fft_dma->fft_control_base_addr, getpagesize());
  close(fft_dma->fft_control_fd);
  // printf("[ INFO] Un-map of virtual address obtained to fft control slave ...\n");
  // printf("[ INFO] Closing file descriptor to fft control slave...\n");
}
