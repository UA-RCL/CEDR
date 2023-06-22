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
    unsigned int tx_status = 0;
    unsigned int tx_ctr = 0;
    while ( (dma_control_base_addr[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02){
      if(tx_ctr % 10000000 == 0){
        tx_status = dma_control_base_addr[DMA_OFFSET_MM2S_STATUS];
        printf("[dma] DMA %d TX Status: \"0x%x\"\n", 1, tx_status);
      }
      tx_ctr++;
    }
}
int data;
//###################################################################################
// Function to Check if DMA RX to complete
//###################################################################################
void dma_wait_for_rx_complete() {
  unsigned int ctr = 0;
  //unsigned int status_low = 0;
  //unsigned int status_high = 0;
  unsigned int status = 0;

  while ((dma_control_base_addr[DMA_OFFSET_S2MM_STATUS] & 0x0003) != 0x0002) {
    if (ctr % 10000000 == 0) {
      //status_low = dma_control_base_addr[DMA_OFFSET_S2MM_STATUS];
      //status_high = dma_control_base_addr[DMA_OFFSET_S2MM_STATUS + 1];
      status = dma_control_base_addr[DMA_OFFSET_S2MM_STATUS];
      // TODO: Check the DMA IP user guide to make sure this is giving useful info
      printf("[dma] DMA %d RX Status: \"0x%x\"\n", 1, status);
    }
    ctr++;
  }
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
// Function to Reset DMA
//###################################################################################
void reset_dma() {
    printf("[ DEBUG] Writing on TX Control\n");
    dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x4);
    printf("[ DEBUG] Writing on RX Control\n");
    dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x4);
    printf("[ DEBUG] Waiting for TX to be idle\n");
    dma_wait_for_tx_idle();
    printf("[ DEBUG] Waiting for RX to be idle\n");
    dma_wait_for_rx_idle();
}


//###################################################################################
// Function to obtain pointers to udmabuf
//###################################################################################
float* init_udmabuf(int fd_udmabuf) {


    fd_udmabuf = open("/dev/udmabuf0", O_RDWR|O_SYNC);
    if(fd_udmabuf < 0) {
      printf("[ERROR] Can't open /dev/udmabuf0. Exiting ...\n");
      exit(1);
    }
    
    printf("[ INFO] Successfully opened /dev/udmabuf0 ...\n");
    
    float *base_addr = (float *)mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_SHARED, fd_udmabuf, 0);

    if(base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to udmabuf buffer. Exiting ...\n");
       exit(1);
    }
    
    printf("[ INFO] Successfully obtained virtual address to udmabuf buffer ...\n");

    int fd_udmabuf_addr = open("/sys/class/u-dma-buf/udmabuf0/phys_addr", O_RDONLY);
    if(fd_udmabuf_addr < 0) {
      printf("[ERROR] Can't open /sys/class/u-dma-buf/udmabuf0/phys_addr. Exiting ...\n");
      exit(1);
    } 
  
    printf("[ INFO] Successfully opened /sys/class/u-dma-buf/udmabuf0/phys_addr ...\n");
    read(fd_udmabuf_addr, attr, 1024);
    sscanf(attr, "%lx", &udmabuf_phys_addr); 
    close(fd_udmabuf_addr);

    // Reset DMA
    printf("[ INFO] Resetting DMA\n");
    reset_dma();
    printf("[ INFO] Successfully reset DMA\n");
    
    return base_addr;
    
}

//###################################################################################
// Function to Initiate Transfer of Matrix over DMA
//###################################################################################
void setup_tx () {

    dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x3);
    dma_write_reg(DMA_OFFSET_MM2S_SRCLWR,  udmabuf_phys_addr);
    dma_write_reg(DMA_OFFSET_MM2S_LENGTH,  (DIM * 4 * 2));
    //dma_wait_for_tx_complete();
    printf("[ INFO] Sent data to HW accelerator over DMA ...\n");
}

//###################################################################################
// Function to Initiate RX over DMA
//###################################################################################
void setup_rx () {

    dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x3);
    dma_write_reg(DMA_OFFSET_S2MM_SRCLWR,  (udmabuf_phys_addr + (DIM * 4 * 2)));
    dma_write_reg(DMA_OFFSET_S2MM_LENGTH,  (DIM * 4 * 2));
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

