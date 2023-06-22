#include "dma.h"
#include <unistd.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define STATUS_INTERVAL 1000
#define WAIT_LIMIT (10 * (STATUS_INTERVAL))

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(volatile unsigned int* base, unsigned int offset, unsigned int data) { 
  LOG("[dma] Writing data: 0x%08x to address: %p\n", data, (base + offset));
  *(base + offset) = data; 
}

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_tx(volatile unsigned int* dma_config_addr, unsigned int source_addr, unsigned int num_bytes) {
  dma_write_reg(dma_config_addr, DMA_OFFSET_MM2S_CONTROL, 0x3);
  dma_write_reg(dma_config_addr, DMA_OFFSET_MM2S_SRCLWR, source_addr);
  dma_write_reg(dma_config_addr, DMA_OFFSET_MM2S_LENGTH, num_bytes);
}

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx(volatile unsigned int* dma_config_addr, unsigned int dest_addr, unsigned int num_bytes) {
  dma_write_reg(dma_config_addr, DMA_OFFSET_S2MM_CONTROL, 0x3);
  dma_write_reg(dma_config_addr, DMA_OFFSET_S2MM_SRCLWR, dest_addr);
  dma_write_reg(dma_config_addr, DMA_OFFSET_S2MM_LENGTH, num_bytes);
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(volatile unsigned int* base) {
  unsigned int ctr = 0;

  while ((base[DMA_OFFSET_MM2S_STATUS] & 0x01) != 0x01) {
    if (ctr % (STATUS_INTERVAL) == 0) {
      LOG("[dma] DMA at %p waiting for TX (MM2S) idle\n", base);
    }
    if (ctr == WAIT_LIMIT) {
      LOG("[dma] DMA at %p giving up waiting for TX (MM2S) idle!\n", base);
      return;
    }
    ctr++;
  }
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(volatile unsigned int* base) {
  unsigned int ctr = 0;

  while ((base[DMA_OFFSET_S2MM_STATUS] & 0x01) != 0x01) {
    if (ctr % (STATUS_INTERVAL) == 0) {
      LOG("[dma] DMA at %p waiting for RX (S2MM) idle\n", base);
    }
    if (ctr == WAIT_LIMIT) {
      LOG("[dma] DMA at %p giving up waiting for RX (S2MM) idle!\n", base);
      return;
    }
    ctr++;
  }
}

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(volatile unsigned int* base) {
  unsigned int ctr = 0;
  unsigned int status = 0;

  while ((base[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02) {
    if (ctr % (STATUS_INTERVAL) == 0) {
      status = base[DMA_OFFSET_MM2S_STATUS];
      LOG("[dma] DMA at %p TX Status: \"0x%x\"\n", base, status);
    }
    if (ctr == WAIT_LIMIT) {
      LOG("[dma] DMA at %p giving up waiting for TX (MM2S) completion!\n", base);
      return;
    }
    ctr++;
  }
}

//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(volatile unsigned int* base) {
  unsigned int ctr = 0;
  unsigned int status = 0;

  while ((base[DMA_OFFSET_S2MM_STATUS] & 0x03) != 0x02) {
    if (ctr % (STATUS_INTERVAL) == 0) {
      status = base[DMA_OFFSET_S2MM_STATUS];
      LOG("[dma] DMA at %p RX Status: \"0x%x\"\n", base, status);
    }
    if (ctr == WAIT_LIMIT) {
      LOG("[dma] DMA at %p giving up waiting for RX (S2MM) completion!\n", base);
      return;
    }
    ctr++;
  }
}

void reset_dma(volatile unsigned int* base) {
  LOG("Resetting DMA at base address: %p\n", base);
  dma_write_reg(base, DMA_OFFSET_MM2S_CONTROL, 0x4);
  dma_write_reg(base, DMA_OFFSET_S2MM_CONTROL, 0x4);
  dma_wait_for_tx_idle(base);
  dma_wait_for_rx_idle(base);
}

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
volatile unsigned int* init_dma(unsigned int base_addr) {
  LOG("[dma] Initializing DMA with control base address 0x%x\n", base_addr);
  // Open device memory in order to get access to DMA control slave
  int fd;
  volatile unsigned int* addr;

  fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[dma] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  // Obtain virtual address to DMA control slave through mmap
  addr = (volatile unsigned int *) mmap(nullptr,
                                       getpagesize(),
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED,
                                       fd,
                                       base_addr);

  if (addr == MAP_FAILED) {
    LOG("[dma] Can't obtain memory map to DMA control slave. Exiting ...\n");
    exit(1);
  } else {
    LOG("[dma] mmap succeeded with result: %p\n", (unsigned int*) addr);
  }

  // As per mmap(2):
  // "After the mmap() call has returned, the file descriptor, fd, can be closed immediately without invalidating the mapping."
  close(fd);

  return addr;
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma(volatile unsigned int* dma_control_addr) {
  munmap((unsigned int*) dma_control_addr, getpagesize());
}

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf(uint32_t buf_num, size_t udmabuf_size, volatile unsigned int** virtual_addr_result, uint64_t* phys_addr_result) {
  int fd, phys_addr_fd;
  volatile unsigned int* virtual_addr = nullptr;
  uint64_t phys_addr = 0;
  char udmabuf_name[10];
  char udmabuf_dev_path[100];
  char udmabuf_sys_path[100];
  char *attr = (char*) calloc(1024, sizeof(char));

  snprintf(udmabuf_name, sizeof(udmabuf_name)/sizeof(udmabuf_name[0]), "udmabuf%d", buf_num);
  snprintf(udmabuf_dev_path, sizeof(udmabuf_dev_path)/sizeof(udmabuf_dev_path[0]), "/dev/udmabuf%d", buf_num);
  snprintf(udmabuf_sys_path, sizeof(udmabuf_dev_path)/sizeof(udmabuf_dev_path[0]), "/sys/class/u-dma-buf/udmabuf%d/phys_addr", buf_num);

  LOG("[dma] udmabuf name: %s, udmabuf dev path: %s, udmabuf sys path: %s\n", udmabuf_name, udmabuf_dev_path, udmabuf_sys_path);

  LOG("[dma] Opening file descriptor to /dev/%s\n", udmabuf_name);
  fd = open(udmabuf_dev_path, O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[dma] Can't open %s. Exiting ...\n", udmabuf_dev_path);
    exit(1);
  }

  virtual_addr = (volatile unsigned int*) mmap(nullptr,
                                               udmabuf_size,
                                               PROT_READ | PROT_WRITE,
                                               MAP_SHARED,
                                               fd,
                                               0);
  close(fd);

  if (virtual_addr == MAP_FAILED) {
    LOG("[dma] Can't obtain memory map to %s buffer (is the buffer too small?). Exiting ...\n", udmabuf_name);
    exit(1);
  }

  // phys_addr_fd = open(udmabuf_sys_path, O_RDONLY);
  // if (phys_addr_fd < 0) {
  //   fprintf(stderr, "[dma] Can't open %s. Exiting...\n", udmabuf_sys_path);
  //   exit(1);
  // }

  // read(phys_addr_fd, attr, 1024);
  // sscanf(attr, "%lx", &phys_addr);
  // close(phys_addr_fd);

  if ((phys_addr_fd  = open(udmabuf_sys_path, O_RDONLY)) != -1) {
    read(phys_addr_fd, attr, 1024);
    sscanf(attr, "%lx", &phys_addr);
    close(phys_addr_fd);
  } else {
    fprintf(stderr, "[dma] Can't open %s. Exiting...\n", udmabuf_sys_path);
    exit(1);
  }

  free(attr);

  *virtual_addr_result = virtual_addr;
  *phys_addr_result = phys_addr;
}

//###################################################################################
// Function to remove memory maps to udmabuf
//###################################################################################
void close_udmabuf(volatile unsigned int* udmabuf_base_addr, size_t udmabuf_size) {
  munmap((unsigned int*) udmabuf_base_addr, udmabuf_size);
}
