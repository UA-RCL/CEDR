#pragma once

#include <fcntl.h>
#include <cstdint>

// #define __DASH_DMA_DEBUG__

#ifdef LOG
#undef LOG
#endif

#ifdef __DASH_DMA_DEBUG__
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
#endif

//###################################################################################
// DMA Register Space Definitions
//###################################################################################

// Control Registers of DMA Memory Mapped to Stream Interface
#define DMA_OFFSET_MM2S_CONTROL 0
#define DMA_OFFSET_MM2S_STATUS  1
#define DMA_OFFSET_MM2S_SRCLWR  6
#define DMA_OFFSET_MM2S_SRCUPR  7
#define DMA_OFFSET_MM2S_LENGTH  10
// Control Registers of DMA Stream to Memory Mapped Interface
#define DMA_OFFSET_S2MM_CONTROL 12
#define DMA_OFFSET_S2MM_STATUS  13
#define DMA_OFFSET_S2MM_SRCLWR  18
#define DMA_OFFSET_S2MM_SRCUPR  19
#define DMA_OFFSET_S2MM_LENGTH  22

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(volatile unsigned int *base, unsigned int offset, unsigned int data);

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_tx(volatile unsigned int *dma_config_addr, unsigned int source_addr, unsigned int num_bytes);

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx(volatile unsigned int *dma_config_addr, unsigned int dest_addr, unsigned int num_bytes);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(volatile unsigned int *base);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(volatile unsigned int *base);

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(volatile unsigned int *base);

//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(volatile unsigned int *base);

//###################################################################################
// Function to reset DMA
//###################################################################################
void reset_dma(volatile unsigned int *base);

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
volatile unsigned int* init_dma(unsigned int base_addr);

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma(volatile unsigned int* dma_control_addr);

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf(uint32_t buf_num, size_t udmabuf_size, volatile unsigned int** virtual_addr_result, uint64_t* phys_addr_result);

//###################################################################################
// Function to remove memory maps to udmabuf
//###################################################################################
void close_udmabuf(volatile unsigned int* udmabuf_virtual_addr, size_t udmabuf_size);

