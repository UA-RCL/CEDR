/*
 * Note: this image is built for Han's demo platform from July 2020
 */
#include        <cstdio>
#include        <cstdlib>
#include        <fcntl.h>
#include        <string.h>
#include        <time.h>
#include        <sys/types.h>
#include        <sys/mman.h>
#include        <fftw3.h>
#include        <math.h>
#include        <time.h>

#include      "fft_dma.h"

#define fft_dim_size 2048
#define SEC2NANOSEC 1000000000

#ifdef ARM
static fft_dma_cfg fft_dma_0, fft_dma_1;

void __attribute__((constructor)) setup(void) {
  printf("[fft] Initializing FFT accelerators\n");
  fft_dma_0.list = fft_dma_config::fft_0;
  fft_dma_1.list = fft_dma_config::fft_1;

  // Ignore FFT 0

  // Initialize FFT 1
  init_dma(&fft_dma_1);
  init_udmabuf_1(&fft_dma_1);
  init_fft(&fft_dma_1);
}

void __attribute__((destructor)) teardown(void) {
  printf("[fft] Tearing down FFT accelerators\n");
  // Ignore FFT 0

  // Close FFT 1
  close_dma(&fft_dma_1);
  close_udma_buffer(&fft_dma_1);
  close_fft(&fft_dma_1);
}
#endif

// TODO: Figure out the right parameters here
// TODO: LLVM Params: i64* %23, double** %25, double** %30, double** %7, double** %8, i64* %9, %struct.gsl_fft_complex_wavetable** %10, %struct.gsl_fft_complex_workspace** %11, i64* %12
// TODO:              len,      input,        output,       input,       output,      len,     unused,                                  unused,                                  size_t i
extern "C" void fft2048_cpu(int64_t* len, double** input, double** output, double** input2_unused, double** output2_unused, int64_t *len2_unused, void* struct1_unused, void* struct2_unused, int64_t *i_unused) {
  printf("[fft] FFTW on the CPU for a 2048-Pt FFT\n");
  thread_local fftwf_complex *in, *out;
  thread_local fftwf_plan p;
  thread_local clock_t begin, end;

  thread_local int32_t j;
  in = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * 2048);
  out = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * 2048);
  p = fftwf_plan_dft_1d(2048, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  for (j = 0; j < 4096; j+=2) {
    in[j/2][0] = (*input)[j];
    in[j/2][1] = (*input)[j+1];
  }

  begin = clock();
  fftwf_execute(p);
  end = clock();
  printf("FFTW Time taken: %lf\n", (double)(end-begin)/CLOCKS_PER_SEC);

  for (j = 0; j < 4096; j+=2) {
    (*output)[j] = out[j/2][0];
    (*output)[j+1] = out[j/2][1];
  }

  fftwf_destroy_plan(p);
  fftwf_free(in);
  fftwf_free(out);
  return;
}

#ifdef ARM
// TODO: LLVM Params: i64* %23, double** %25, double** %30, double** %7, double** %8, i64* %9, %struct.gsl_fft_complex_wavetable** %10, %struct.gsl_fft_complex_workspace** %11, i64* %12
// TODO:              len,      input,        output,       input,       output,      len,     unused,                                  unused,                                  size_t i
extern "C" void fft2048_accel(int64_t* len, double** input, double** output, double** input_unused, double** output_unused, int64_t *len2_unused, void* struct1_unused, void* struct2_unused, int64_t *i_unused) {
  thread_local struct timespec start_accel, end_accel;
  printf("[fft] Running a 2048-Pt FFT on FFT 1 on the FPGA\n");

  thread_local TYPE *fft_input_udmabuf_base = (TYPE *)(fft_dma_1.udmabuf_base_addr);
  thread_local TYPE *fft_output_udmabuf_base = (TYPE *)(fft_dma_1.udmabuf_base_addr + fft_dim_size * 2);
  thread_local size_t i;

  //printf("[fft] Configuring FFT 1 for %d pt FFT\n", fft_dim_size);
  fft_dma_1.fft_dim = fft_dim_size;
  config_fft(&fft_dma_1, (unsigned int)log2(fft_dim_size));

  //printf("[fft] Copying input to fpga_input buffer\n");
  for (i = 0; i < fft_dim_size * 2; i++) {
    fft_input_udmabuf_base[i] = (*input)[i];
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_accel);

  //printf("[fft] Calling setup_rx...\n");
  setup_fft_dma_rx(&fft_dma_1);

  //printf("[fft] Calling setup_tx...\n");
  setup_fft_dma_tx(&fft_dma_1);

  //printf("[fft] Waiting for DMA TX to complete...\n");
  dma_wait_for_tx_complete(&fft_dma_1);

  //printf("[fft] Waiting for DMA RX to complete...\n");
  dma_wait_for_rx_complete(&fft_dma_1);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end_accel);

  printf("[fft] FFT execution time (ns): %f\n",
         ((double)end_accel.tv_sec * SEC2NANOSEC + (double)end_accel.tv_nsec) -
         ((double)start_accel.tv_sec * SEC2NANOSEC + (double)start_accel.tv_nsec));

  //printf("[fft] Copying output from fpga_output buffer\n");
  for (i = 0; i < fft_dim_size * 2; i++) {
    (*output)[i] = fft_output_udmabuf_base[i];
  }

  printf("[fft] Finished 2048-Pt FFT Execution on FFT 1 on the FPGA\n");
  return;
}
#endif