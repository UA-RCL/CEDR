#include <complex.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "dash_types.h"

#include "dma.h"
#include "fft.h"

#define SEC2NANOSEC 1000000000

static volatile unsigned int* fft_control_base_addr[NUM_FFTS];
static volatile unsigned int* dma_control_base_addr[NUM_FFTS];
static volatile unsigned int* udmabuf_base_addr;
static uint64_t               udmabuf_phys_addr;

#if defined(FFT_GPIO_RESET_BASE_ADDRS)
static volatile unsigned int* fft_gpio_reset_base_addr[NUM_FFTS];
#endif

void __attribute__((constructor)) setup_fft(void) {
  LOG("[fft] Running FFT constructor\n");

  for (uint8_t i = 0; i < NUM_FFTS; i++) {
#if defined(FFT_GPIO_RESET_BASE_ADDRS)
    LOG("[fft] Initializing FFT's reset GPIO at 0x%x\n", FFT_GPIO_RESET_BASE_ADDRS[i]);
    fft_gpio_reset_base_addr[i] = init_fft_reset(FFT_GPIO_RESET_BASE_ADDRS[i]);
    reset_fft_and_dma(fft_gpio_reset_base_addr[i]);
#endif

    LOG("[fft] Initializing FFT DMA at 0x%x\n", FFT_DMA_CTRL_BASE_ADDRS[i]);
    dma_control_base_addr[i] = init_dma(FFT_DMA_CTRL_BASE_ADDRS[i]);

    LOG("[fft] Initializing FFT Control at 0x%x\n", FFT_CONTROL_BASE_ADDRS[i]);
    fft_control_base_addr[i] = init_fft(FFT_CONTROL_BASE_ADDRS[i]);
  }

  LOG("[fft] Initializing udmabuf %d\n", FFT_UDMABUF_NUM);
  init_udmabuf(FFT_UDMABUF_NUM, FFT_UDMABUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
  LOG("[fft] udmabuf base address is %p\n", udmabuf_base_addr);

  LOG("[fft] FFT constructor complete!\n");
}

void __attribute__((destructor)) teardown_fft(void) {
  LOG("[fft] Running FFT destructor\n");
  close_udmabuf(udmabuf_base_addr, FFT_UDMABUF_SIZE);

  for (uint8_t i = 0; i < NUM_FFTS; i++) {
    close_fft(fft_control_base_addr[i]);
    close_dma(dma_control_base_addr[i]);
#if defined(FFT_GPIO_RESET_BASE_ADDRS)
    close_fft_reset(fft_gpio_reset_base_addr[i]);
#endif
  }

  LOG("[fft] FFT destructor complete!\n");
}

void fft_accel(fft_cmplx_type* input, fft_cmplx_type* output, size_t fft_size, bool isForwardTransform, uint8_t resource_idx) {
  // Check if we're not a power of two
  // Fun trick from the internet: if you're a power of two, your bitmask with yourself-1 is fully 0
  // But it doesn't work if you're 0 so there's a second case
  if (((fft_size & (fft_size - 1)) != 0) || (fft_size == 0)) {
    LOG("[fft-%u] ERROR: Unable to compute FFT of size %lu-Pt as it is not a power of two. Exiting...\n", resource_idx, fft_size);
    return;
  }

  //LOG("[fft-%u] Running a %d-Pt %s on FFT 1 on the FPGA\n", fft_size, isForwardTransform ? "FFT" : "IFFT");
  printf("[fft-%u] Running a %lu-Pt %s on FFT %u on the FPGA\n", resource_idx, fft_size, isForwardTransform ? "FFT" : "IFFT", resource_idx);
  struct timespec start_accel {};
  struct timespec end_accel {};
  size_t i;

  volatile unsigned int *fft_control_base = fft_control_base_addr[resource_idx];
  volatile unsigned int *dma_control_base = dma_control_base_addr[resource_idx];
  volatile unsigned int *udmabuf_base = udmabuf_base_addr + (resource_idx * (UDMABUF_PARTITION_SIZE / sizeof(unsigned int)));
  uint64_t udmabuf_phys = udmabuf_phys_addr + (resource_idx * UDMABUF_PARTITION_SIZE);

  LOG("[fft-%u] fft control reg is %p\n", resource_idx, fft_control_base);
  LOG("[fft-%u] dma control reg is %p\n", resource_idx, dma_control_base);
  LOG("[fft-%u] udmabuf base address is %p\n", resource_idx, udmabuf_base);
  LOG("[fft-%u] udmabuf phys address is 0x%lx\n", resource_idx, udmabuf_phys);

#if defined(FFT_GPIO_RESET_BASE_ADDRS)
  volatile unsigned int *fft_gpio_reset_base = fft_gpio_reset_base_addr[resource_idx];
  LOG("[fft-%u] fft gpio reset base address is %p\n", resource_idx, fft_gpio_reset_base);
  LOG("[fft-%u] Using GPIO to reset both FFT and FFT DMA IP cores...\n", resource_idx);
  reset_fft_and_dma(fft_gpio_reset_base);
#else
  LOG("[fft-%u] Resetting DMA engine\n", resource_idx);
  reset_dma(dma_control_base);
#endif

  LOG("[fft-%u] Configuring as %lu-Pt %s\n", resource_idx, fft_size, isForwardTransform ? "FFT" : "IFFT");
  if (isForwardTransform) {
    config_fft(fft_control_base, log2(fft_size));
  } else {
    config_ifft(fft_control_base, log2(fft_size));
  }

  LOG("[fft-%u] Copying input buffer to udmabuf (udmabuf_base: %p, input: %p)\n", resource_idx, udmabuf_base, input);
  memcpy((unsigned int*) udmabuf_base, input, fft_size * sizeof(fft_cmplx_type));

  LOG("[fft-%u] Calling setup_rx\n", resource_idx);
  setup_rx(dma_control_base, udmabuf_phys + (fft_size * sizeof(fft_cmplx_type)), fft_size * sizeof(fft_cmplx_type));

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_accel);

  LOG("[fft-%u] Calling setup_tx\n", resource_idx);
  setup_tx(dma_control_base, udmabuf_phys, fft_size * sizeof(fft_cmplx_type));
  
  LOG("[fft-%u] Waiting for RX to complete\n", resource_idx);
  dma_wait_for_rx_complete(dma_control_base);
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_accel);

  LOG("[fft-%u] %lu-Pt FFT accelerator execution time (ns): %lf\n", resource_idx, fft_size,
         ((double)end_accel.tv_sec * SEC2NANOSEC + (double)end_accel.tv_nsec) - ((double)start_accel.tv_sec * SEC2NANOSEC + (double) start_accel.tv_nsec));

  LOG("[fft-%u] Memcpy output back from physical address: %lx\n", resource_idx, udmabuf_phys + (2 * fft_size * sizeof(unsigned int)));
  memcpy(output, (unsigned int*) &udmabuf_base[2 * fft_size], fft_size * sizeof(fft_cmplx_type));

  LOG("[fft-%u] Finished %lu-Pt %s Execution on FFT %u on the FPGA\n", resource_idx, fft_size, isForwardTransform ? "FFT" : "IFFT", resource_idx);
}

extern "C" void DASH_FFT_flt_fft(dash_cmplx_flt_type** input, dash_cmplx_flt_type** output, size_t* size, bool* isForwardTransform, uint8_t resource_idx) {
  // TODO: This doesn't actually mean the types are the same. The fft_re_type might be integral still :(
  // Additional checks can be done here using either std::is_same if we drop baseline C support (C++ only) or look into C11's "_Generic"
  if (sizeof(dash_re_flt_type) == sizeof(fft_re_type)) {
    // Assume that everything is fine for us to just pass forward the user's buffer without creating a separate conversion copy
    fft_accel((fft_cmplx_type*) (*input), (fft_cmplx_type*) (*output), *size, *isForwardTransform, resource_idx);
    if((*isForwardTransform) == false){
      for (size_t i = 0; i < (*size); i++) { 
        (*output)[i].re = (dash_re_flt_type) (*output)[i].re / (*size);
        (*output)[i].im = (dash_re_flt_type) (*output)[i].im / (*size);
      }
    }
  }
  // If their sizes aren't the same, we know for sure that there needs to be some kind of type conversion, though
  else {
    fft_cmplx_type* inp = (fft_cmplx_type*) malloc((*size) * sizeof(fft_cmplx_type));
    fft_cmplx_type* out = (fft_cmplx_type*) malloc((*size) * sizeof(fft_cmplx_type));
  
    for (size_t i = 0; i < (*size); i++) {
      inp[i].re = (fft_re_type) (*input)[i].re;
      inp[i].im = (fft_re_type) (*input)[i].im;
    }

    fft_accel(inp, out, *size, *isForwardTransform, resource_idx);

    if(*isForwardTransform){
      for (size_t i = 0; i < (*size); i++) {
        (*output)[i].re = (dash_re_flt_type) out[i].re;
        (*output)[i].im = (dash_re_flt_type) out[i].im;
      }
    }
    else{
      for (size_t i = 0; i < (*size); i++) {
        (*output)[i].re = (dash_re_flt_type) out[i].re / (*size);
        (*output)[i].im = (dash_re_flt_type) out[i].im / (*size);
      }
    }

    free(inp);
    free(out);
  }
}

extern "C" void DASH_FFT_int_fft(dash_cmplx_int_type** input, dash_cmplx_int_type** output, size_t* size, bool* isForwardTransform, uint8_t resource_idx) {
  // TODO: This doesn't actually mean the types are the same. The fft_re_type might be floating point still :(
  // Additional checks can be done here using either std::is_same if we drop baseline C support (C++ only) or look into C11's "_Generic"
  if (sizeof(dash_re_int_type) == sizeof(fft_re_type)) {
    // Assume that everything is fine for us to just pass forward the user's buffer without creating a separate conversion copy
    fft_accel((fft_cmplx_type*) (*input), (fft_cmplx_type*) (*output), *size, *isForwardTransform, resource_idx);
    if(!(*isForwardTransform)){
      for (size_t i = 0; i < (*size); i++) { 
        (*output)[i].re = (dash_re_int_type) (*output)[i].re / (*size);
        (*output)[i].im = (dash_re_int_type) (*output)[i].im / (*size);
      }
    }
  }
  // If their sizes aren't the same, we know for sure that there needs to be some kind of type conversion, though
  else {
    fft_cmplx_type* inp = (fft_cmplx_type*) malloc((*size) * sizeof(fft_cmplx_type));
    fft_cmplx_type* out = (fft_cmplx_type*) malloc((*size) * sizeof(fft_cmplx_type));
  
    for (size_t i = 0; i < (*size); i++) {
      inp[i].re = (fft_re_type) (*input)[i].re;
      inp[i].im = (fft_re_type) (*input)[i].im;
    }

    fft_accel(inp, out, *size, *isForwardTransform, resource_idx);

    if(*isForwardTransform){
      for (size_t i = 0; i < (*size); i++) {
        (*output)[i].re = (dash_re_int_type) out[i].re;
        (*output)[i].im = (dash_re_int_type) out[i].im;
      }
    }
    else{
      for (size_t i = 0; i < (*size); i++) {
        (*output)[i].re = (dash_re_int_type) out[i].re / (*size);
        (*output)[i].im = (dash_re_int_type) out[i].im / (*size);
      }
    }

    free(inp);
    free(out);
  }
}

#if defined(__FFT_ENABLE_MAIN)
typedef enum fft_input_type_t {
  delta,
  constant,
  centered_sinusoid,
  offset_sinusoid
} fft_input_type;

uint32_t _check_fft_result_(dash_cmplx_flt_type *fft_actual, dash_cmplx_flt_type *fft_expected, size_t FFT_DIM) {
  int error_count = 0;
  dash_re_flt_type diff;
  dash_cmplx_flt_type c, d;

  printf("[fft] Checking actual versus expected output\n");
  for (size_t i = 0; i < FFT_DIM; i++) {
    c = fft_expected[i];
    d = fft_actual[i];

    diff = std::abs(std::sqrt(c.re*c.re) - std::sqrt(d.re*d.re)) / std::sqrt(c.re*c.re);

    if (diff > 0.0001) {
      fprintf(stderr, "[fft] [ERROR] Expected = %f %fi, Actual FFT = %f %fi, index = %ld\n", c.re, c.im, d.re, d.im, i);
      error_count++;
    }
  }

  if (error_count == 0) {
    printf("[fft] FFT Passed!\n");
    return 0;
  } else {
    printf("[fft] FFT Failed!\n");
    return 1;
  }
}

void _generate_fft_test_values_(dash_cmplx_flt_type *fft_input, dash_cmplx_flt_type *fft_expected, size_t FFT_SIZE, fft_input_type inp_type, bool is_forward_transform) {
  const size_t f_bin = (inp_type == centered_sinusoid) ? FFT_SIZE / 2 :
                       (inp_type == offset_sinusoid) ? FFT_SIZE / 4 : 0;
  dash_re_flt_type* t_arr;

  switch (FFT_SIZE) {
  case 16:
  case 32:
  case 64:
  case 128:
  case 256:
  case 512:
  case 1024:
  case 2048:
    switch (inp_type) {
      case delta:
        // The input is a delta function 
        fft_input[0].re = 1;
        fft_input[0].im = 0;
        // As such, the expected output is the constant 1 function
        for (size_t i = 0; i < FFT_SIZE; i++) {
          fft_expected[i].re = (is_forward_transform) ? 1 : 1.0/FFT_SIZE;
          fft_expected[i].im = 0;
        }
        break;
      case constant:
        // The input is the constant "1" function
        for (size_t i = 0; i < FFT_SIZE; i++) {
          fft_input[i].re = 1;
          fft_input[i].im = 0;
        }
        // And as such, the output is a delta function with height == FFT_SIZE
        fft_expected[0].re = (is_forward_transform) ? FFT_SIZE : 1;
        fft_expected[0].im = 0       ;
        break;
      case centered_sinusoid:
      case offset_sinusoid:
        // Create a linspaced input from 0 to 2*pi (forgetting the last point)
        t_arr = (dash_re_flt_type *) calloc(FFT_SIZE, sizeof(dash_re_flt_type));
        for (size_t i = 0; i < FFT_SIZE; i++) {
          t_arr[i] = (((dash_re_flt_type) i) / FFT_SIZE);
        }
        // Create a complex sinusoid that will perfectly match with bin "f_bin"
        for (size_t i = 0; i < FFT_SIZE; i++) {
          fft_input[i].re = cos(2 * M_PI * f_bin * t_arr[i]);
          fft_input[i].im = sin(2 * M_PI * f_bin * t_arr[i]);

          // If we're computing a forward transform (FFT), then this sinusoid will fall in the "f_bin"th bin of the real part of the resulting array
          // Otherwise, it will fall in the negative version of that bin mod FFT_SIZE. aka FFT_SIZE - f_bin
          if (((i == f_bin) && is_forward_transform) || ((i == FFT_SIZE - f_bin) && !is_forward_transform)) {
            fft_expected[i].re = (is_forward_transform) ? FFT_SIZE : 1;
            fft_expected[i].im = 0;
          } else {
            fft_expected[i].re = 0;
            fft_expected[i].im = 0;
          }
        }
        free(t_arr);
        break;
      default:
        // Can't reach unless someone adds an enum value and doesn't update the case statement >:(
        break;
    }
    break;
  default:
    fprintf(stderr, "[fft] Unknown FFT_SIZE specified in _generate_fft_test_values_: %lu\n", FFT_SIZE);
    exit(1);
  }
}

void _print_fft_result_(dash_cmplx_flt_type *fft_result, size_t FFT_SIZE, bool isForwardTransform) {
  printf("[fft] Printing received %lu-Pt %s output\n", FFT_SIZE, isForwardTransform ? "FFT" : "IFFT");
  for (size_t i = 0; i < FFT_SIZE; i++) {
    printf("(%lf + %lfi)", fft_result[i].re, fft_result[i].im);
    if (i < FFT_SIZE - 1) {
      printf(", ");
    }
  }
  printf("\n");
}

uint32_t _run_fft_tests_(size_t FFT_SIZE, fft_input_type inp_type, bool is_forward_transform, uint8_t fft_num, bool exit_on_failure, bool print_intermediate_results) {
  dash_cmplx_flt_type *fft_input, *fft_expected, *fft_actual_accel;
  const char* FFT_TYPE = is_forward_transform ? "FFT" : "IFFT";

  uint32_t tests_failed = 0;

  fft_input = (dash_cmplx_flt_type *) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  fft_expected = (dash_cmplx_flt_type *) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  fft_actual_accel = (dash_cmplx_flt_type *) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));

  printf("[fft] Generating input data and expected output\n");
  _generate_fft_test_values_(fft_input, fft_expected, FFT_SIZE, inp_type, is_forward_transform);

  printf("[fft] Computing %lu-Pt FFT using the accelerator kernel\n", FFT_SIZE);
  //fft_accel(fft_input, fft_actual_accel, FFT_SIZE, is_forward_transform);
  DASH_FFT_flt_fft(&fft_input, &fft_actual_accel, &FFT_SIZE, &is_forward_transform, fft_num);

  if (print_intermediate_results) {
    _print_fft_result_(fft_actual_accel, FFT_SIZE, is_forward_transform);
  }

  printf("[fft] Checking the accelerator output against the reference expected values\n");
  tests_failed += _check_fft_result_(fft_actual_accel, fft_expected, FFT_SIZE);

  if (exit_on_failure && tests_failed > 0) {
    exit(1);
  }

  printf("[fft] %lu-Pt %s tests complete!\n", FFT_SIZE, FFT_TYPE);

  free(fft_input);
  free(fft_expected);
  free(fft_actual_accel);

  return tests_failed;
}

int main(int argc, char** argv) {
  // Stores the cumulative total number of tests that have failed
  uint32_t tests_failed = 0;
  // How many did we run tho?
  uint32_t tests_run = 0;
  // If true, exit upon the first failed test ("fail fast")
  bool exit_on_failure = false;
  // If true, print full FFT output from each implementation as it runs
  bool print_intermediate_outputs = false;

  uint32_t fft_tests_failed = 0;
  uint32_t ifft_tests_failed = 0;

  for (uint8_t fft_num = 0; fft_num < NUM_FFTS; fft_num++) {
    for (int8_t forward_transform = 1; forward_transform >= 0; forward_transform--) { 
      // Looking at the Xilinx IP configuration, we can support any power of two from 16pt to 2048pt atm
      for (size_t fft_size = 16; fft_size <= 2048; fft_size *= 2) {
        printf("[fft] Beginning %lu-Pt %s tests on accelerator %u\n", fft_size, forward_transform ? "FFT" : "IFFT", fft_num);
        // Sparse tests
        fft_tests_failed += _run_fft_tests_(fft_size, delta, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
        // Dense tests
        fft_tests_failed += _run_fft_tests_(fft_size, constant, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
        // Centered sinusoidal tests
        fft_tests_failed += _run_fft_tests_(fft_size, centered_sinusoid, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
        // Offset sinusoidal tests
        fft_tests_failed += _run_fft_tests_(fft_size, offset_sinusoid, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
        tests_run += 4;
      }
    }
  }

  // Alternate main function that allows the user to specify which exact transform they would like to use
  // if (argc > 1) {
  //   size_t fft_size;
  //   fft_input_type type;
  //   bool fwd_transform;

  //   printf("[fft] Parsing custom parameters\n");

  //   fft_size = (size_t) strtoul(argv[1], nullptr, 10);
  //   type = (fft_input_type) strtoul(argv[2], nullptr, 10);
  //   fwd_transform = (bool) strtoul(argv[3], nullptr, 10);

  //   printf("[fft] Running a single transform with parameters: %u, %d, %s\n", fft_size, type, fwd_transform ? "FFT" : "IFFT");
  //   fft_tests_failed += _run_fft_tests_(fft_size, type, fwd_transform, exit_on_failure, print_intermediate_outputs);
  // } else {
  //   fft_tests_failed += _run_fft_tests_(32, delta, true, exit_on_failure, print_intermediate_outputs);
  // }
  

  tests_failed = fft_tests_failed + ifft_tests_failed;

  printf("[fft] In total, %d FFT/IFFT tests failed (out of %d tests!)\n", tests_failed, tests_run);
  return (tests_failed > 0) ? 1 : 0;
}

// Alternate main function useful for testing concurrency issues by spawning a thread to test each FFT at the same time
/*
void* test_accelerator(void* arg) {
  uint8_t fft_num = *((uint8_t*) arg);

  // Stores the cumulative total number of tests that have failed
  uint32_t tests_failed = 0;
  // How many did we run tho?
  uint32_t tests_run = 0;
  // If true, exit upon the first failed test ("fail fast")
  bool exit_on_failure = false;
  // If true, print full FFT output from each implementation as it runs
  bool print_intermediate_outputs = false;

  uint32_t fft_tests_failed = 0;
  uint32_t ifft_tests_failed = 0;

  for (int8_t forward_transform = 1; forward_transform >= 0; forward_transform--) { 
    // Looking at the Xilinx IP configuration, we can support any power of two from 16pt to 2048pt atm
    for (size_t fft_size = 16; fft_size <= 2048; fft_size *= 2) {
      printf("[fft-%u] Beginning %u-Pt %s tests\n", fft_num, fft_size, forward_transform ? "FFT" : "IFFT");
      // Sparse tests
      fft_tests_failed += _run_fft_tests_(fft_size, delta, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
      // Dense tests
      fft_tests_failed += _run_fft_tests_(fft_size, constant, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
      // Centered sinusoidal tests
      fft_tests_failed += _run_fft_tests_(fft_size, centered_sinusoid, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
      // Offset sinusoidal tests
      fft_tests_failed += _run_fft_tests_(fft_size, offset_sinusoid, forward_transform, fft_num, exit_on_failure, print_intermediate_outputs);
      tests_run += 4;
    }
  }

  tests_failed = fft_tests_failed + ifft_tests_failed;

  printf("[fft-%u] In total, %d FFT/IFFT tests failed (out of %d tests!)\n", fft_num, tests_failed, tests_run);

  int* result = (int*) malloc(sizeof(int));
  *result = tests_failed;
  return result;
}

int main(void) {
  uint8_t args[NUM_FFTS];
  void* results[NUM_FFTS];
  pthread_t threads[NUM_FFTS];
  int total_tests_failed = 0;

  for (uint8_t i = 0; i < NUM_FFTS; i++) {
    args[i] = i;
    pthread_create(&threads[i], nullptr, test_accelerator, (void *) &(args[i]));
  }

  for (uint8_t i = 0; i < NUM_FFTS; i++) {
    pthread_join(threads[i], &(results[i]));
    int tests_failed = *((int*) results[i]);
    total_tests_failed += tests_failed;
    free(((int*) results[i]));
  }

  printf("[fft] Across both accelerators, %d tests failed\n", total_tests_failed);
  return (total_tests_failed > 0) ? 1 : 0;
}
*/
#endif
