#pragma once
#define FFT1_CONTROL_BASE_ADDR 0xA0001000
#define FFT2_CONTROL_BASE_ADDR 0xA0003000

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void init_fft1();

void init_fft2();

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void fft_write_reg(unsigned int *base, unsigned int offset, int data);

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_ifft(unsigned int *base, unsigned int size);

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_fft(unsigned int *base, unsigned int size);

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_fft1();

void close_fft2();
