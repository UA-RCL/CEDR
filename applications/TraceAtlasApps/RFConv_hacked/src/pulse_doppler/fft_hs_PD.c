#include <stdio.h>
#include <math.h>
#include <complex.h>

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

#include "dma.h"
#include "fft_hwa.h"

typedef double complex cplx;
 
void fft_hs_PD(int fft_id, float* fft_in, float* fft_out, int n) {

    int i;
    float        *udmabuf_base_addr;
    unsigned int *dma_control_base_addr;
    unsigned int *fft_control_base_addr;
    unsigned int udmabuf_phys_addr;

    struct timespec start1, end1;
    float exec_time;

    if (fft_id == 1) {
        udmabuf_base_addr     = udmabuf1_base_addr;
        dma_control_base_addr = dma1_control_base_addr;
        udmabuf_phys_addr     = udmabuf1_phys_addr;
        fft_control_base_addr = fft1_control_base_addr;
    } else {
        udmabuf_base_addr     = udmabuf2_base_addr;
        dma_control_base_addr = dma2_control_base_addr;
        udmabuf_phys_addr     = udmabuf2_phys_addr;
        fft_control_base_addr = fft2_control_base_addr;
    }

    config_fft(fft_control_base_addr, log2(n));

    memcpy(udmabuf_base_addr, fft_in, sizeof(float) * n * 2);

    // Setup RX over DMA
    setup_rx(dma_control_base_addr, udmabuf_phys_addr, n);

    // Transfer Matrix A over the DMA
    setup_tx(dma_control_base_addr, udmabuf_phys_addr, n);

    // Wait for DMA to complete transfer to destination buffer
    dma_wait_for_rx_complete(dma_control_base_addr);

    memcpy(fft_out, &udmabuf_base_addr[n * 2], sizeof(float) * n * 2);

}

void ifft_hs_PD(int fft_id, float* fft_in, float* fft_out, int n) {

    int i, n2;
    float        *udmabuf_base_addr;
    unsigned int *dma_control_base_addr;
    unsigned int *fft_control_base_addr;
    unsigned int udmabuf_phys_addr;

    if (fft_id == 1) {
        udmabuf_base_addr     = udmabuf1_base_addr;
        dma_control_base_addr = dma1_control_base_addr;
        udmabuf_phys_addr     = udmabuf1_phys_addr;
        fft_control_base_addr = fft1_control_base_addr;
    } else {
        udmabuf_base_addr     = udmabuf2_base_addr;
        dma_control_base_addr = dma2_control_base_addr;
        udmabuf_phys_addr     = udmabuf2_phys_addr;
        fft_control_base_addr = fft2_control_base_addr;
    }

    config_ifft(fft_control_base_addr, log2(n));

    memcpy(udmabuf_base_addr, fft_in, sizeof(float) * n * 2);

    // Setup RX over DMA
    setup_rx(dma_control_base_addr, udmabuf_phys_addr, n);

    // Transfer Matrix A over the DMA
    setup_tx(dma_control_base_addr, udmabuf_phys_addr, n);

    // Wait for DMA to complete transfer to destination buffer
    dma_wait_for_rx_complete(dma_control_base_addr);

    memcpy(fft_out, &udmabuf_base_addr[n * 2], sizeof(float) * n * 2);

    //for(int i = 0; i < n * 2; i++) {
    //    fft_out[i] = fft_out[i] / n;
    //}

}
