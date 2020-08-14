
//###################################################################################
// DMA Register Space Definitions
//###################################################################################

// Base Address of DMA AXI Slave in Platform Design
#define DMA_CONTROL_BASE_ADDR    0xA0005000
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

//###################################################################################
// Function to Write Data to DMA Control Register
//###################################################################################
void dma_write_reg(unsigned int offset, int data) {

    dma_control_base_addr[offset] = data;
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_tx_idle() {
    
    while ( (dma_control_base_addr[DMA_OFFSET_MM2S_STATUS] & 0x01) != 0x01);
}

//###################################################################################
// Function to Check if DMA Idle
//###################################################################################
void dma_wait_for_rx_idle() {
    
    while ( (dma_control_base_addr[DMA_OFFSET_S2MM_STATUS] & 0x01) != 0x01);
}

//###################################################################################
// Function to Check if DMA TX to complete
//###################################################################################
void dma_wait_for_tx_complete() {
    printf("dma_wait_for_tx_complete DMA_OFFSET_MM2S_STATUS: %X\n", dma_control_base_addr[DMA_OFFSET_MM2S_STATUS]);
    
    while ( (dma_control_base_addr[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02);
}

//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete() {
        printf("dma_wait_for_rx_complete DMA_OFFSET_S2MM_STATUS: %X\n", dma_control_base_addr[DMA_OFFSET_S2MM_STATUS]);

    while ( (dma_control_base_addr[DMA_OFFSET_S2MM_STATUS] & 0x03) != 0x02);
}

//###################################################################################
// Function to initialize memory maps to DMA 
//###################################################################################
void init_dma() {

    // Open device memory in order to get access to DMA control slave
    int dma_control_fd = open("/dev/mem", O_RDWR|O_SYNC);
    if(dma_control_fd < 0) {
      printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
      exit(1);
    }

    printf("[ INFO] Successfully opened /dev/mem ...\n");

    // Obtain virtual address to DMA control slave through mmap
    dma_control_base_addr = (unsigned int*) mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, dma_control_fd, DMA_CONTROL_BASE_ADDR);

    if(dma_control_base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to DMA control slave. Exiting ...\n");
       exit(1);
    }

    printf("[ INFO] Successfully obtained virtual address to DMA control slave ...\n");
}

    
//###################################################################################
// Function to obtain pointers to udmabuf0
//###################################################################################
TYPE* init_udmabuf0(int fd_udmabuf0) {

    // udmabuf0
    fd_udmabuf0 = open("/dev/udmabuf2", O_RDWR|O_SYNC);
    if(fd_udmabuf0 < 0) {
      printf("[ERROR] Can't open /dev/udmabuf2. Exiting ...\n");
      exit(1);
    }
    
    printf("[ INFO] Successfully opened /dev/udmabuf2 ...\n");
    
    TYPE *base_addr = (TYPE *)mmap(NULL, NUM_BYTES_TO_TRANSFER, PROT_READ|PROT_WRITE, MAP_SHARED, fd_udmabuf0, 0);

    if(base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to udmabuf2 buffer. Exiting ...\n");
       exit(1);
    }
    
    //printf("[ INFO] Successfully obtained virtual address to udmabuf2 buffer ...\n");

    int fd_udmabuf_addr = open("/sys/class/udmabuf/udmabuf2/phys_addr", O_RDONLY);
    if(fd_udmabuf_addr < 0) {
      printf("[ERROR] Can't open /sys/class/udmabuf/udmabuf2/phys_addr. Exiting ...\n");
      exit(1);
    } 
  
    //printf("[ INFO] Successfully opened /sys/class/udmabuf/udmabuf2/phys_addr ...\n");
    read(fd_udmabuf_addr, attr, 1024);
    sscanf(attr, "%lx", &udmabuf0_phys_addr); 
    close(fd_udmabuf_addr);

    return base_addr;
    
}

//###################################################################################
// Function to obtain pointers to udmabuf1
//###################################################################################
TYPE* init_udmabuf1(int fd_udmabuf1) {

    // udmabuf1
    fd_udmabuf1 = open("/dev/udmabuf1", O_RDWR|O_SYNC);
    if(fd_udmabuf1 < 0) {
      printf("[ERROR] Can't open /dev/udmabuf1. Exiting ...\n");
      exit(1);
    }
    
    printf("[ INFO] Successfully opened /dev/udmabuf1 ...\n");
    
    TYPE *base_addr = (TYPE *)mmap(NULL, NUM_BYTES_TO_TRANSFER, PROT_READ|PROT_WRITE, MAP_SHARED, fd_udmabuf1, 0);

    if(base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to udmabuf1 buffer. Exiting ...\n");
       exit(1);
    }
    
    printf("[ INFO] Successfully obtained virtual address to udmabuf1 buffer ...\n");

    int fd_udmabuf_addr = open("/sys/class/udmabuf/udmabuf1/phys_addr", O_RDONLY);
    if(fd_udmabuf_addr < 0) {
      printf("[ERROR] Can't open /sys/class/udmabuf/udmabuf1/phys_addr. Exiting ...\n");
      exit(1);
    } 
  
    printf("[ INFO] Successfully opened /sys/class/udmabuf/udmabuf1/phys_addr ...\n");
    read(fd_udmabuf_addr, attr, 1024);
    sscanf(attr, "%lx", &udmabuf1_phys_addr); 
    close(fd_udmabuf_addr);

    return base_addr;
    
}

//###################################################################################
// Function to Generate Inputs
//###################################################################################
void gen_inputs(TYPE *base_addr, TYPE *ref_output) {
    
    int data = 0;
    for (int i = 0; i < NUM_BYTES_TO_TRANSFER / (4*2); i++) {
        
        // Generate a random number between 0 and 50
        data = rand() % 50; 

        // Put the values in udmabuf source buffer
        base_addr[i*2] = 1;
        base_addr[i*2+1] = 2;

        // Store the values in reference output buffer
        ref_output[i] = i;

        //printf("[ INFO] Input number [%02d]: %d\n", (i + 1), base_addr[i]);
    }
    base_addr[NUM_BYTES_TO_TRANSFER / 4-1] = 0;
}


//###################################################################################
// Function to clear address
//###################################################################################
void clear_address(TYPE *base_addr, TYPE *ref_output) {
    
    for (int i = 0; i < NUM_BYTES_TO_TRANSFER / 4; i++) {
 
        // Put the values in udmabuf source buffer
        base_addr[i] = 0;
    }

}

//###################################################################################
// Function to Check Outputs from HW
//###################################################################################
void check_outputs(TYPE *base_addr, TYPE *ref_output) {
    
    int error_count = 0;

    for (int i = 977; i < 993; i++) {
        
        if (base_addr[i] != ref_output[i]) {
            error_count++;
            printf("[ERROR] Value from DMA at index [%02d] INCORRECT. Reference: [%02d], HW: [%02d]\n", (i + 1), ref_output[i], base_addr[i]);
        } else {
            printf("[ INFO] Value from DMA at index [%02d]   correct. Reference: [%02d], HW: [%02d]\n", (i + 1), ref_output[i], base_addr[i]);
        }
    }

    if (error_count == 0) {
        printf("\n[ INFO] Transactions over DMA PASSED! ...\n");
    } else {
        printf("\n[ INFO] Transactions over DMA FAILED! ...\n");
    }
}

//###################################################################################
// Function to Initiate Transfer of Data over DMA
//###################################################################################
void setup_tx () {

    dma_write_reg(DMA_OFFSET_MM2S_SRCLWR,  udmabuf0_phys_addr);
    dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x3);
    dma_write_reg(DMA_OFFSET_MM2S_LENGTH,  NUM_BYTES_TO_TRANSFER);
    dma_wait_for_tx_complete();
    printf("[ INFO] Sent data to HW accelerator over DMA ...\n");
}

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx () {

    dma_write_reg(DMA_OFFSET_S2MM_SRCLWR,  udmabuf1_phys_addr);
    dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x3);
    dma_write_reg(DMA_OFFSET_S2MM_LENGTH,  NUM_BYTES_TO_TRANSFER);
    printf("[ INFO] Initiated DMA receiver ...\n");
}

//###################################################################################
// Function to remove memory maps to DMA 
//###################################################################################
void close_dma() {
    munmap(dma_control_base_addr, getpagesize());
    close(dma_control_fd);
    //printf("[ INFO] Un-map of virtual address obtained to DMA control slave ...\n");
    //printf("[ INFO] Closing file descriptor to DMA control slave...\n");
}
