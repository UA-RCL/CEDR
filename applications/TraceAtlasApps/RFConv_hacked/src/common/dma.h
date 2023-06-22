//###################################################################################
// DMA Register Space Definitions
//###################################################################################

// Base Address of DMA AXI Slave in Platform Design
#define DMA1_CONTROL_BASE_ADDR 0xA0000000
#define DMA2_CONTROL_BASE_ADDR 0xA0004000
// Control Registers of DMA Memory Mapped to Stream Interface
#define DMA_OFFSET_MM2S_CONTROL 0
#define DMA_OFFSET_MM2S_STATUS 1
#define DMA_OFFSET_MM2S_SRCLWR 6
#define DMA_OFFSET_MM2S_SRCUPR 7
#define DMA_OFFSET_MM2S_LENGTH 10
// Control Registers of DMA Stream to Memory Mapped Interface
#define DMA_OFFSET_S2MM_CONTROL 12
#define DMA_OFFSET_S2MM_STATUS 13
#define DMA_OFFSET_S2MM_SRCLWR 18
#define DMA_OFFSET_S2MM_SRCUPR 19
#define DMA_OFFSET_S2MM_LENGTH 22

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(unsigned int *base, unsigned int offset, int data);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle(unsigned int *base);

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle(unsigned int *base);

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete(unsigned int *base);

//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete(unsigned int *base);

//###################################################################################
// Function to initialize memory maps to DMA
//###################################################################################
void init_dma1();
void init_dma2();

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_tx(unsigned int *base, unsigned int udmabuf_phys_addr, unsigned int n);

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx(unsigned int *base, unsigned int udmabuf_phys_addr, unsigned int n);

//###################################################################################
// Function to remove memory maps to DMA
//###################################################################################
void close_dma1();
void close_dma2();

//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
void init_udmabuf1();
void init_udmabuf2();
