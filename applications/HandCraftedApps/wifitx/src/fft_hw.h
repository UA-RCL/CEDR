#define FFT_CONTROL_BASE_ADDR 0xA0001000

//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void init_fft() {

	// Open device memory in order to get access to DMA control slave
    int fft_control_fd = open("/dev/mem", O_RDWR|O_SYNC);
    if(fft_control_fd < 0) {
      printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
      exit(1);
    }

    //printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
    fft_control_base_addr = (unsigned int*) mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fft_control_fd, FFT_CONTROL_BASE_ADDR);

    if(fft_control_base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to FFT control slave. Exiting ...\n");
       exit(1);
    }

    //printf("[ INFO] Successfully obtained virtual address to FFT control slave ...\n");
}

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void fft_write_reg(unsigned int offset, int data) {

    fft_control_base_addr[offset] = data;
}

//###################################################################################
// Function to initialize memory maps to FFT 
//###################################################################################
void config_ifft() {
    
    fft_write_reg(0x0, 0x0);
    //printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to initialize memory maps to FFT 
//###################################################################################
void config_fft() {
    
    fft_write_reg(0x0, 0x1);
    //printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function - Check Result of FFT
//###################################################################################
void check_result(comp_t sw[], float *hw) {


    int error_count = 0;
    float diff;
    float c, d;
    
    for (int i = 0; i < DIM * 2; i++) {
        if (i % 2 == 0) {
            c = sw[i / 2].real;
        } else {
            c = sw[i / 2].imag;
        }
        d = *(hw + i);
        diff = abs(c - d) / c * 100;

        if (diff > 0.01) {
            printf("[ERROR] Ref = %f, HW = %f\n", c, d);
            printf("[ERROR] Error in result at index [%d]. Error: %f ...\n", (i + 1), diff);
            error_count++;
        } else {
            //printf("[ INFO] Actual = %.3f, Ref = %.3f\n", d, c);
        }
    }

    if (error_count == 0) {
        //printf("\n[ INFO] FFT PASSED!!\n\n");
    } else {
        printf("\n[ERROR] FFT FAILED!!\n\n");
    }

}
