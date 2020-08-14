/*
 * fft_dma.h
 *
 *  Created on: Jun 25, 2020
 *      Author: hanguang yu
 */

#ifndef FFT_DMA_H_
#define FFT_DMA_H_

// Base Address of DMA AXI Slave in Platform Design
#define DMA0_CONTROL_BASE_ADDR    0xA0000000
#define FFT0_CONTROL_BASE_ADDR    0xA0003000
#define DMA1_CONTROL_BASE_ADDR    0xA0001000
#define FFT1_CONTROL_BASE_ADDR    0xA0004000
// Control Registers of DMA Memory Mapped to Stream Interface
#define DMA_OFFSET_MM2S_CONTROL  0
#define DMA_OFFSET_MM2S_STATUS   1
#define DMA_OFFSET_MM2S_SRCLWR   6
#define DMA_OFFSET_MM2S_SRCUPR   7
#define DMA_OFFSET_MM2S_LENGTH   10
// Control Registers of DMA Stream to Memory Mapped Interface
#define DMA_OFFSET_S2MM_CONTROL  12
#define DMA_OFFSET_S2MM_STATUS   13
#define DMA_OFFSET_S2MM_SRCLWR   18
#define DMA_OFFSET_S2MM_SRCUPR   19
#define DMA_OFFSET_S2MM_LENGTH   22
#define udma_buffer_size		1048576           //bytes

#define MAX_FFT_DIM     2048

#define TYPE    float

typedef struct fft_dma_config{

	int			dma_control_fd;
	volatile unsigned int *dma_control_base_addr;

	int           fft_control_fd;
	volatile unsigned int *fft_control_base_addr;
	unsigned int         fft_dim;

	int			  fd_udmabuf;
	unsigned long long  udmabuf_phys_addr;
	volatile unsigned int *udmabuf_base_addr;

	enum fft_dma_list {fft_0 , fft_1} list;
}fft_dma_cfg;

//static fft_dma_cfg fft_dma_0, fft_dma_1;


//enum fft_dma_list {fft_dma_0 = &fft_dma_0, fft_dma_1 = &fft_dma_1};

//void dma_write_reg(fft_dma_cfg *fft_dma, unsigned int offset, int data);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(fft_dma_cfg *fft_dma);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(fft_dma_cfg *fft_dma) ;

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(fft_dma_cfg *fft_dma) ;
//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(fft_dma_cfg *fft_dma) ;

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
void init_dma(fft_dma_cfg *fft_dma);

void init_fft(fft_dma_cfg *fft_dma);

//void init_dma_1(fft_dma_cfg *fft_dma);

//void init_fft_1(fft_dma_cfg *fft_dma);

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf_0(fft_dma_cfg *fft_dma);

void init_udmabuf_1(fft_dma_cfg *fft_dma);

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_fft_dma_tx (fft_dma_cfg *fft_dma);

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_fft_dma_rx (fft_dma_cfg *fft_dma);

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma(fft_dma_cfg *fft_dma);

//###################################################################################
// Function to remove memory maps to udma buffer
//###################################################################################
void close_udma_buffer(fft_dma_cfg *fft_dma);

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
//void fft_write_reg(fft_dma_cfg *fft_dma, unsigned int offset, int data);

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_ifft(fft_dma_cfg *fft_dma, unsigned int size);

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void config_fft(fft_dma_cfg *fft_dma, unsigned int size);
//###################################################################################
// Function to remove memory maps to fft controller
//###################################################################################
void close_fft(fft_dma_cfg *fft_dma);


//void gen_input_fft (TYPE*fft_input, TYPE *fft_input_ref);

//void gen_ref_result (TYPE *fft_output_ref);

void check_result(TYPE *sw, TYPE *hw, fft_dma_cfg *fft_dma);

#endif /* FFT_DMA_H_ */
