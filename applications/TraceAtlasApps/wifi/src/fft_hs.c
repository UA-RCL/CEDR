#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "baseband_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <math.h>
#include <time.h>

#include "common.h"

#define SEC2NANOSEC  1000000000
extern int           dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int           fd_udmabuf1;
extern int           fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern unsigned int  udmabuf1_phys_addr;
extern TYPE         *udmabuf1_base_addr;
extern int           dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int           fd_udmabuf2;
extern int           fft2_control_fd;
extern unsigned int *fft2_control_base_addr;
extern unsigned int  udmabuf2_phys_addr;
extern TYPE         *udmabuf2_base_addr;

int           dma1_control_fd;
unsigned int *dma1_control_base_addr;
int           fd_udmabuf1;
int           fft1_control_fd;
unsigned int *fft1_control_base_addr;
unsigned int  udmabuf1_phys_addr;
TYPE         *udmabuf1_base_addr;
int           dma2_control_fd;
unsigned int *dma2_control_base_addr;
int           fd_udmabuf2;
int           fft2_control_fd;
unsigned int *fft2_control_base_addr;
unsigned int  udmabuf2_phys_addr;
TYPE         *udmabuf2_base_addr;

float        *udmabuf_base_addr;
unsigned int *dma_control_base_addr;
unsigned int udmabuf_phys_addr;

#include "dma.h"
#include "fft_hwa.h"

double PI = 3.141593;
typedef double complex cplx;
 
void _fft_hs(cplx buf[], cplx out[], int n, int step) {

    int i;

    if (step < n) {
        _fft_hs(out, buf, n, step * 2);
        _fft_hs(out + step, buf + step, n, step * 2);
        for (i = 0; i < n; i += 2 * step) {
	    cplx t = cexp(-I * PI * i / n) * out[i + step];
	    buf[i / 2]     = out[i] + t;
	    buf[(i + n)/2] = out[i] - t;
        }
    }
}

void fft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy) {

    int i;
    cplx out[n], buf[n];
    float fft_hw[DIM * 2];

    for (i = 0; i < n; i++) buf[i] = (double)fdata[i].real + (double)fdata[i].imag * I;
    for (i = 0; i < n; i++) out[i] = buf[i];

    if (hw_fft_busy == 1) {

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] FFT running on A53\n");
        #endif

        _fft_hs(buf, out, n, 1);

        for (i = 0; i < n; i++) {
            fdata[i].real = (float)creal(buf[i]);
            fdata[i].imag = (float)cimag(buf[i]);;
        }
    } else {

        if (fft_id == 1) {
            udmabuf_base_addr     = udmabuf1_base_addr;
            dma_control_base_addr = dma1_control_base_addr;
            udmabuf_phys_addr     = udmabuf1_phys_addr;
        } else {
            udmabuf_base_addr     = udmabuf2_base_addr;
            dma_control_base_addr = dma2_control_base_addr;
            udmabuf_phys_addr     = udmabuf2_phys_addr;
        }

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        struct timespec start1, end1;
        float exec_time;
        printf("[INFO] FFT running on accelerator\n");
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        #endif
        
        memcpy(udmabuf_base_addr, fdata, sizeof(float) * DIM * 2);

        // Setup RX over DMA
        setup_rx(dma_control_base_addr, udmabuf_phys_addr);

        // Transfer Matrix A over the DMA
        setup_tx(dma_control_base_addr, udmabuf_phys_addr);

        //dma_wait_for_tx_complete();

        // Wait for DMA to complete transfer to destination buffer
        dma_wait_for_rx_complete(dma_control_base_addr);

        memcpy(fft_hw, &udmabuf_base_addr[DIM * 2], sizeof(float) * DIM * 2);

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
        exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
        printf("[INFO] RX-FFT execution time (ns): %f\n", exec_time);
        #endif

        for (i = 0; i < n; i++) {
            fdata[i].real = (float) fft_hw[(i * 2)];
            fdata[i].imag = (float) fft_hw[(i * 2) + 1];
        }
        
        // Compare SW and HW results
        // check_result(fdata, fft_hw);

    }
}

void ifft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy) {

    int i, n2;
    cplx out[n], buf[n];
    cplx tmp;
    float fft_hw[DIM * 2];

    for (i = 0; i < n; i++) buf[i] = (double)fdata[i].real + (double)fdata[i].imag * I;
    for (i = 0; i < n; i++) out[i] = buf[i];

    if (hw_fft_busy == 1) {

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] IFFT running on A53\n");
        #endif

        _fft_hs(buf, out, n, 1);

        n2 = n/2;
        buf[0] = buf[0]/n;
        buf[n2] = buf[n2]/n;
        for(i=1; i<n2; i++) {
          tmp = buf[i]/n;
          buf[i] = buf[n-i]/n;
          buf[n-i] = tmp;
        }

        for (i = 0; i < n; i++) {
            fdata[i].real = (float)creal(buf[i]);
            fdata[i].imag = (float)cimag(buf[i]);;
        }
        
    } else {

        if (fft_id == 1) {
            udmabuf_base_addr     = udmabuf1_base_addr;
            dma_control_base_addr = dma1_control_base_addr;
            udmabuf_phys_addr     = udmabuf1_phys_addr;
        } else {
            udmabuf_base_addr     = udmabuf2_base_addr;
            dma_control_base_addr = dma2_control_base_addr;
            udmabuf_phys_addr     = udmabuf2_phys_addr;
        }

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        struct timespec start1, end1;
        float exec_time;
        printf("[INFO] IFFT running on accelerator\n");
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        #endif
        
        memcpy(udmabuf_base_addr, fdata, sizeof(float) * DIM * 2);

        // Setup RX over DMA
        setup_rx(dma_control_base_addr, udmabuf_phys_addr);

        // Transfer Matrix A over the DMA
        setup_tx(dma_control_base_addr, udmabuf_phys_addr);

        //dma_wait_for_tx_complete();

        // Wait for DMA to complete transfer to destination buffer
        dma_wait_for_rx_complete(dma_control_base_addr);

        memcpy(fft_hw, &udmabuf_base_addr[DIM * 2], sizeof(float) * DIM * 2);

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
        exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
        printf("[INFO] TX-IFFT execution time (ns): %f\n", exec_time);
        #endif

        for (i = 0; i < n; i++) {
            fdata[i].real = (float) fft_hw[(i * 2)] / 128;
            fdata[i].imag = (float) fft_hw[(i * 2) + 1] / 128;
        }

        // Compare SW and HW results
        //check_result(fdata, fft_hw);

    }
   
}
