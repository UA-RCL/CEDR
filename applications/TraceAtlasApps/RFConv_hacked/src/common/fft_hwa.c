#include "fft_hwa.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common.h"

extern int fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern float *udmabuf1_base_addr;
extern int fft2_control_fd;
extern unsigned int *fft2_control_base_addr;
extern float *udmabuf2_base_addr;

int fft1_control_fd;
unsigned int *fft1_control_base_addr;
float *udmabuf1_base_addr;
int fft2_control_fd;
unsigned int *fft2_control_base_addr;
float *udmabuf2_base_addr;

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void init_fft1() {
	// Open device memory in order to get access to DMA control slave
	fft1_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fft1_control_fd < 0) {
		printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
	fft1_control_base_addr = (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fft1_control_fd,
	                                              FFT1_CONTROL_BASE_ADDR);

	if (fft1_control_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to FFT1 control slave. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to FFT1 control slave ...\n");
}

void init_fft2() {
	// Open device memory in order to get access to DMA control slave
	fft2_control_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fft2_control_fd < 0) {
		printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
	fft2_control_base_addr = (unsigned int *)mmap(0, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fft2_control_fd,
	                                              FFT2_CONTROL_BASE_ADDR);

	if (fft2_control_base_addr == MAP_FAILED) {
		printf("[ERROR] Can't obtain memory map to FFT2 control slave. Exiting ...\n");
		exit(1);
	}

	// printf("[ INFO] Successfully obtained virtual address to FFT2 control slave ...\n");
}

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void fft_write_reg(unsigned int *base, unsigned int offset, int data) { *(base + offset) = data; }

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_ifft(unsigned int *base, unsigned int size) {
	fft_write_reg(base, 0x0, size);
	// printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_fft(unsigned int *base, unsigned int size) {
	fft_write_reg(base, 0x0, (0x1 << 8 | size));
	// printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_fft1() {
	munmap(fft1_control_base_addr, getpagesize());
	close(fft1_control_fd);
	// printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
	// printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}

void close_fft2() {
	munmap(fft2_control_base_addr, getpagesize());
	close(fft2_control_fd);
	// printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
	// printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}
