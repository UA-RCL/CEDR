#include "dma.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "common.h"

extern int dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int fd_udmabuf1;
extern float *udmabuf1_base_addr;
extern unsigned int udmabuf1_phys_addr;

extern int dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int fd_udmabuf2;
extern float *udmabuf2_base_addr;
extern unsigned int udmabuf2_phys_addr;

int dma1_control_fd;
unsigned int *dma1_control_base_addr;
int fd_udmabuf1;
float *udmabuf1_base_addr;
unsigned int udmabuf1_phys_addr;

int dma2_control_fd;
unsigned int *dma2_control_base_addr;
int fd_udmabuf2;
float *udmabuf2_base_addr;
unsigned int udmabuf2_phys_addr;

static char attr[1024];

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(unsigned int *base, unsigned int offset, int data) { *(base + offset) = data; }

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(unsigned int *base) {
	while ((*(base + DMA_OFFSET_MM2S_STATUS) & 0x01) != 0x01)
		;
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(unsigned int *base) {
	while ((*(base + DMA_OFFSET_S2MM_STATUS) & 0x01) != 0x01)
		;
}

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(unsigned int *base) {
	while ((*(base + DMA_OFFSET_MM2S_STATUS) & 0x03) != 0x02)
		;
}

//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(unsigned int *base) {
	while ((*(base + DMA_OFFSET_S2MM_STATUS) & 0x03) != 0x02)
		;
}

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
void init_dma1() {
	// Open device memory in order to get access to DMA control slave
	dma1_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (dma1_control_fd < 0) {
		printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
	dma1_control_base_addr = (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, dma1_control_fd,
	                                              DMA1_CONTROL_BASE_ADDR);

	if (dma1_control_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to DMA1 control slave. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to DMA1 control slave ...\n");
}

void init_dma2() {
	// Open device memory in order to get access to DMA control slave
	dma2_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (dma2_control_fd < 0) {
		printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
	dma2_control_base_addr = (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, dma2_control_fd,
	                                              DMA2_CONTROL_BASE_ADDR);

	if (dma2_control_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to DMA2 control slave. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to DMA2 control slave ...\n");
}

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf1() {
	fd_udmabuf1 = open("/dev/udmabuf0", O_RDWR | O_SYNC);
	if (fd_udmabuf1 < 0) {
		printf("[ERROR] Can't open /dev/udmabuf0. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/udmabuf0 ...\n");

	udmabuf1_base_addr = (float *)mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd_udmabuf1, 0);

	if (udmabuf1_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to udmabuf0 buffer. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to udmabuf buffer ...\n");

	int fd_udmabuf_addr = open("/sys/class/udmabuf/udmabuf0/phys_addr", O_RDONLY);
	if (fd_udmabuf_addr < 0) {
		printf("[ERROR] Can't open /sys/class/udmabuf/udmabuf0/phys_addr. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /sys/class/udmabuf/udmabuf0/phys_addr ...\n");
	read(fd_udmabuf_addr, attr, 1024);
	sscanf(attr, "%x", &udmabuf1_phys_addr);
	close(fd_udmabuf_addr);

	// Reset DMA
	dma_write_reg(dma1_control_base_addr, DMA_OFFSET_MM2S_CONTROL, 0x4);
	dma_write_reg(dma1_control_base_addr, DMA_OFFSET_S2MM_CONTROL, 0x4);
	dma_wait_for_tx_idle(dma1_control_base_addr);
	dma_wait_for_rx_idle(dma1_control_base_addr);
}

void init_udmabuf2() {
	fd_udmabuf2 = open("/dev/udmabuf1", O_RDWR | O_SYNC);
	if (fd_udmabuf2 < 0) {
		printf("[ERROR] Can't open /dev/udmabuf1. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/udmabuf1 ...\n");

	udmabuf2_base_addr = (float *)mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd_udmabuf2, 0);

	if (udmabuf2_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to udmabuf1 buffer. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to udmabuf buffer ...\n");

	int fd_udmabuf_addr = open("/sys/class/udmabuf/udmabuf1/phys_addr", O_RDONLY);
	if (fd_udmabuf_addr < 0) {
		printf("[ERROR] Can't open /sys/class/udmabuf/udmabuf1/phys_addr. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /sys/class/udmabuf/udmabuf1/phys_addr ...\n");
	read(fd_udmabuf_addr, attr, 1024);
	sscanf(attr, "%x", &udmabuf2_phys_addr);
	close(fd_udmabuf_addr);

	// Reset DMA
	dma_write_reg(dma2_control_base_addr, DMA_OFFSET_MM2S_CONTROL, 0x4);
	dma_write_reg(dma2_control_base_addr, DMA_OFFSET_S2MM_CONTROL, 0x4);
	dma_wait_for_tx_idle(dma2_control_base_addr);
	dma_wait_for_rx_idle(dma2_control_base_addr);
}

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_tx(unsigned int *base, unsigned int udmabuf_phys_addr, unsigned int n) {
	dma_write_reg(base, DMA_OFFSET_MM2S_SRCLWR, udmabuf_phys_addr);
	dma_write_reg(base, DMA_OFFSET_MM2S_CONTROL, 0x3);
	dma_write_reg(base, DMA_OFFSET_MM2S_LENGTH, (n * 4 * 2));
	// dma_wait_for_tx_complete();
	// printf("[ INFO] Sent data to HW accelerator over DMA ...\n");
}

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx(unsigned int *base, unsigned int udmabuf_phys_addr, unsigned int n) {
	dma_write_reg(base, DMA_OFFSET_S2MM_SRCLWR, (udmabuf_phys_addr + (n * 4 * 2)));
	dma_write_reg(base, DMA_OFFSET_S2MM_CONTROL, 0x3);
	dma_write_reg(base, DMA_OFFSET_S2MM_LENGTH, (n * 4 * 2));
	// printf("Value = %d\n", n * 4 * 2);
	// printf("Value = %d\n", *(base + DMA_OFFSET_S2MM_LENGTH));
	// printf("[ INFO] Initiated DMA receiver ...\n");
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma1() {
	munmap(dma1_control_base_addr, getpagesize());
	close(dma1_control_fd);
	// printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
	// printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}

void close_dma2() {
	munmap(dma2_control_base_addr, getpagesize());
	close(dma2_control_fd);
	// printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
	// printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}
