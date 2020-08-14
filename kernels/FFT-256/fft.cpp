#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fftw3.h>
#include <complex.h>
#include "dma.hpp"
#include "fft_hwa.hpp"

// Arguments:
// i64* i/loop iterator,
// i64* dftSize,
// i32* row,
// i32* col,
// double** dftMatrix,
// double** c,
// double** X1

//void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p) {
//  for(size_t i = 0; i < 2*n_elements; i+=2)
//  {
//    in[i/2][0] = input_array[i];
//    in[i/2][1] = input_array[i+1];
//  }
//  fftwf_execute(p);
//  for(size_t i = 0; i < 2*n_elements; i+=2)
//  {
//    output_array[i] = (float) out[i/2][0];
//    output_array[i+1] = (float) out[i/2][1];
//  }
//}

extern int          fft1_control_fd;
extern unsigned int *fft1_control_base_addr;

extern unsigned int *dma1_control_base_addr;

extern unsigned int *udmabuf1_base_addr;
extern unsigned int udmabuf1_phys_addr;

extern int          fft2_control_fd;
extern unsigned int *fft2_control_base_addr;

extern unsigned int *dma2_control_base_addr;

extern unsigned int *udmabuf2_base_addr;
extern unsigned int udmabuf2_phys_addr;

#ifdef ARM
void __attribute__((constructor)) setup(void) {
  printf("[fft] Initializing DMA buffers...\n");

  // Virtual Address to DMA Control Slave
  init_dma1();

  // Virtual Address to udmabuf Buffer
  init_fft1();
  init_udmabuf1();
  printf("[fft] DMA, FFT, and udmabuf1 initialized...\n");
}

void __attribute__((destructor)) teardown(void) {
  printf("[fft] Closing DMA buffers...\n");
  close_fft1();
  close_dma1();
  printf("[fft] Complete\n");
}
#endif

extern "C" void fft256_cpu(int64_t* i, int64_t* dftSize, int32_t* row, int32_t* col, double** dftMatrix, double** c, double** X1) {
//  fftwf_fft(c, range_detect_param->in_xcorr1, range_detect_param->out_xcorr1,  range_detect_param->X1, len, range_detect_param->p1)
  printf("[fft] Running a 256-Pt FFT with FFTW on the CPU\n");
  thread_local fftwf_complex *in, *out;
  thread_local fftwf_plan p;
  thread_local clock_t begin, end;

  thread_local int32_t j;
  in = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * 256);
  out = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * 256);
  p = fftwf_plan_dft_1d(256, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  for (j = 0; j < 512; j+=2) {
    in[j/2][0] = (*c)[j];
    in[j/2][1] = (*c)[j+1];
  }

  begin = clock();
  fftwf_execute(p);
  end = clock();
  printf("[fft] Finished 256-Pt FFT with FFTW. Execution Time taken: %lf\n", (double)(end-begin)/CLOCKS_PER_SEC);

  for (j = 0; j < 512; j+=2) {
    (*X1)[j] = out[j/2][0];
    (*X1)[j+1] = out[j/2][1];
  }

  fftwf_destroy_plan(p);
  fftwf_free(in);
  fftwf_free(out);
  return;
}

#ifdef ARM
extern "C" void fft256_accel(int64_t* i, int64_t* dftSize, int32_t* row, int32_t* col, double** dftMatrix, double** c, double** X1) {
  clock_t func_start, func_end;
  func_start = clock();
  printf("[fft] Running a 256-Pt FFT on FFT 0 on the FPGA\n");

  thread_local unsigned int *udmabuf_base_addr;
  thread_local unsigned int *dma_control_base_addr;
  thread_local unsigned int *fft_control_base_addr;
  thread_local unsigned int udmabuf_phys_addr;

  thread_local float *fpga_input, *fpga_output;

  fpga_input = (float*) calloc(512, sizeof(float));
  fpga_output = (float*) calloc(512, sizeof(float));
  int32_t j;
  clock_t begin, end;
  //clock_t begin_dma, end_dma, begin_tx, end_tx;

  for (j = 0; j < 512; j++) {
    fpga_input[j] = (float) (*c)[j];
  }

  udmabuf_base_addr = udmabuf1_base_addr;
  dma_control_base_addr = dma1_control_base_addr;
  udmabuf_phys_addr = udmabuf1_phys_addr;
  fft_control_base_addr = fft1_control_base_addr;

  config_fft(fft_control_base_addr, log2(256));

  memcpy(udmabuf_base_addr, fpga_input, sizeof(float) * 256 * 2);

  // Setup RX over DMA
  setup_rx(dma_control_base_addr, udmabuf_phys_addr, 256);

  // Transfer Matrix A over the DMA
  begin = clock();
  setup_tx(dma_control_base_addr, udmabuf_phys_addr, 256);

  // Wait for DMA to complete transfer to destination buffer
  dma_wait_for_rx_complete(dma_control_base_addr);
  end = clock();

  memcpy(fpga_output, &udmabuf_base_addr[256 * 2], sizeof(float) * 256 * 2);

  for (j = 0; j < 512; j++) {
    (*X1)[j] = (double) fpga_output[j];
  }

  free(fpga_input); free(fpga_output);
  func_end = clock();
  printf("[fft] Finished 256-Pt FFT execution with FFT 0 on FPGA (accelerator execution roughly took: %f, full function roughly took: %f)\n", (double)(end-begin)/CLOCKS_PER_SEC, (double)(func_end-func_start)/CLOCKS_PER_SEC);
  return;
}
#endif