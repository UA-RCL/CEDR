#include        <stdio.h>
#include        <stdint.h>
#include        <stdlib.h>
#include        <fcntl.h>
#include        <string.h>
#include        <time.h>
#include        <sys/types.h>
#include        <sys/mman.h>
#include        <cmath>
#include        <time.h>
#include        <unistd.h>

// Include File Containing DMA and udmabuf Initialization Routines
#include      "dma.h"

// Include File Containing GEMM/Matrix Transfer Routines
#include      "gemm.h"

#define SEC2NANOSEC 1000000000

static volatile unsigned int* dma_control_base_addr[NUM_GEMMS];
static volatile unsigned int* udmabuf_base_addr;
static uint64_t               udmabuf_phys_addr;

void __attribute__((constructor)) setup_gemm(void) {
  LOG("[gemm] Running constructor\n");

  for (uint8_t i = 0; i < NUM_GEMMS; i++) {
    LOG("[gemm] Initializing GEMM DMA at 0x%x\n", GEMM_DMA_CTRL_BASE_ADDRS[i]);
    dma_control_base_addr[i] = init_dma(GEMM_DMA_CTRL_BASE_ADDRS[i]);
    reset_dma(dma_control_base_addr[i]);
  }
  
  LOG("[gemm] Initializing GEMM udmabuf\n");
  init_udmabuf(GEMM_UDMABUF_NUM, GEMM_UDMABUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
  
  LOG("[gemm] GEMM initialization complete!\n");
}

void __attribute__((destructor)) teardown_gemm(void) {
  LOG("[gemm] Running destructor\n");
  close_udmabuf(udmabuf_base_addr, GEMM_UDMABUF_SIZE);
  for (uint8_t i = 0; i < NUM_GEMMS; i++) {
    close_dma(dma_control_base_addr[i]);
  }
  LOG("[gemm] Teardown complete!\n");
}

void gemm_fpga_kern(gemm_re_type *A_re, gemm_re_type *A_im, 
                    gemm_re_type *B_re, gemm_re_type *B_im, 
                    gemm_re_type *C_re, gemm_re_type *C_im,
                    uint8_t resource_idx) {

  // Making this an unconditional log because it's really nice to know even when other logs are disabled (log levels what are those?)
  //printf("[gemm] Dispatching to (4x64 * 64x4) GEMM accelerator %u...\n", resource_idx);
  struct timespec func_start;
  struct timespec func_end;
  struct timespec accel_start;
  struct timespec accel_end;

  clock_gettime(CLOCK_MONOTONIC_RAW, &func_start);

  gemm_re_type* A  = A_re;
  gemm_re_type* Ai = A_im;
  gemm_re_type* B  = B_re;
  gemm_re_type* Bi = B_im;
  gemm_re_type* C  = C_re;
  gemm_re_type* Ci = C_im;

  volatile unsigned int *dma_control_base = dma_control_base_addr[resource_idx];
  volatile unsigned int *udmabuf_base = udmabuf_base_addr + (resource_idx * (UDMABUF_PARTITION_SIZE / sizeof(unsigned int)));
  uint64_t udmabuf_phys = udmabuf_phys_addr + (resource_idx * UDMABUF_PARTITION_SIZE);

  volatile gemm_re_type *gemm_input = (volatile gemm_re_type*) &udmabuf_base[0];
  volatile gemm_re_type *gemm_output = (volatile gemm_re_type*) &udmabuf_base[INPUT_DIM];

  thread_local size_t row;
  thread_local size_t col;
  thread_local size_t iterator;

  // Copy inputs to the input udmabuf source buffer
  //printf("[gemm] Copying inputs into udmabuf\n");
  memcpy((gemm_re_type*) &gemm_input[0],                 A,  A_SIZE/2 * sizeof(gemm_re_type));
  memcpy((gemm_re_type*) &gemm_input[A_SIZE/2],          Ai, A_SIZE/2 * sizeof(gemm_re_type));
  memcpy((gemm_re_type*) &gemm_input[A_SIZE],            B,  B_SIZE/2 * sizeof(gemm_re_type));
  memcpy((gemm_re_type*) &gemm_input[A_SIZE + B_SIZE/2], Bi, B_SIZE/2 * sizeof(gemm_re_type));

  // Setup RX of the finished matrix
  LOG("[gemm] Calling setup RX\n");
  setup_rx(dma_control_base,
           udmabuf_phys + (INPUT_DIM * sizeof(gemm_re_type)),
           OUTPUT_DIM * sizeof(gemm_re_type));

  LOG("[gemm] INPUT_DIM is %d, so we need %lu bytes transmitted into the board\n", INPUT_DIM, INPUT_DIM * sizeof(gemm_re_type));
  LOG("[gemm] Calling setup TX\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &accel_start);
  // Start TX of the input matrices to the accelerator
  setup_tx(dma_control_base,
           udmabuf_phys,
           INPUT_DIM * sizeof(gemm_re_type));

  // Wait for the accelerator to compute and for the DMA engine to get us the data back
  LOG("[gemm] Waiting for RX to complete\n");
  dma_wait_for_rx_complete(dma_control_base);
  clock_gettime(CLOCK_MONOTONIC_RAW, &accel_end);

  LOG("[gemm] Memcpy output back (copying %ld bytes in total)\n", C_SIZE * sizeof(gemm_re_type));
  memcpy(C,  (gemm_re_type*) &gemm_output[0],        (C_SIZE / 2) * sizeof(gemm_re_type));
  memcpy(Ci, (gemm_re_type*) &gemm_output[C_SIZE/2], (C_SIZE / 2) * sizeof(gemm_re_type));

  clock_gettime(CLOCK_MONOTONIC_RAW, &func_end);

  LOG("[gemm] Accelerator execution complete! Total function time: %ld ns. Accelerator (TX -> Compute -> RX) time: %ld ns.\n", 
        (func_end.tv_sec * SEC2NANOSEC + func_end.tv_nsec) - (func_start.tv_sec * SEC2NANOSEC + func_start.tv_nsec),
        (accel_end.tv_sec * SEC2NANOSEC + accel_end.tv_nsec) - (accel_start.tv_sec * SEC2NANOSEC + accel_start.tv_nsec));
}

/*
 * Note: now we have two CPU implementations (DASH_GEMM_cpu and this) that could potentially diverge :(
 * At the same time, it would break our libdash architecture strategy to have this gemm module rely on code from the main dash.cpp
 */
void _calculate_gemm_cpu(gemm_re_type *A, gemm_re_type *Ai, gemm_re_type *B, gemm_re_type *Bi, gemm_re_type *C, gemm_re_type *Ci, size_t A_ROW, size_t A_COL, size_t B_COL) {
  gemm_re_type res1, res2, res3, res4;
  gemm_re_type term1, term2, term3, term4;
  
  for (int i = 0; i < A_ROW; i++) {
    for (int j = 0; j < B_COL; j++) {
      res1 = 0; res2 = 0; res3 = 0; res4 = 0;
      // A_COL better equal B_ROWS or the computer gods will be mad at you >:(
      for (int k = 0; k < A_COL; k++) {
        term1 = A[i * A_COL + k] * B[k * B_COL + j];
        res1 += term1;

        term2 = Ai[i * A_COL + k] * Bi[k * B_COL + j];
        res2 += term2;

        term3 = A[i * A_COL + k] * Bi[k * B_COL + j];
        res3 += term3;

        term4 = Ai[i * A_COL + k] * B[k * B_COL + j];
        res4 += term4;
      }
      C[i * B_COL + j] = res1 - res2;
      Ci[i * B_COL + j] = res3 + res4;
    }
  }
}

extern "C" void DASH_GEMM_flt_gemm(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, uint8_t resource_idx) {
  // TODO: if we change the accelerator to support interleaved Re/Im values, then we can potentially optimize this to require no copying
  LOG("[gemm] Calling DASH_GEMM_int_gemm\n");
  gemm_re_type *A_re = (gemm_re_type *) calloc((*Row_A) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *A_im = (gemm_re_type *) calloc((*Row_A) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *B_re = (gemm_re_type *) calloc((*Col_B) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *B_im = (gemm_re_type *) calloc((*Col_B) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *C_re = (gemm_re_type *) calloc((*Row_A) * (*Col_B), sizeof(gemm_re_type));
  gemm_re_type *C_im = (gemm_re_type *) calloc((*Row_A) * (*Col_B), sizeof(gemm_re_type));

  for (size_t i = 0; i < (*Row_A) * (*Col_A); i++) {
    A_re[i] = (gemm_re_type) (*A)[i].re;
    A_im[i] = (gemm_re_type) (*A)[i].im;
  }

  for (size_t i = 0; i < (*Col_A) * (*Col_B); i++) {
    B_re[i] = (gemm_re_type) (*B)[i].re;
    B_im[i] = (gemm_re_type) (*B)[i].im;
  }

  // If we aren't exactly multiplying (4x64 * 64x4) matrices, (i) cry and (ii) fall back to a CPU impl even if it _drastically_ confuses CEDR
  // Also: note that we are comparing the size_t typed values against their #define macro equivalents here
  if (*Row_A != A_ROWS || *Col_A != A_COLS || *Col_B != B_COLS) {
    LOG("[gemm] Falling back to CPU implementation! This gemm operation is not sized correctly to be compatible with our accelerator!\n");
    _calculate_gemm_cpu(A_re, A_im, B_re, B_im, C_re, C_im, *Row_A, *Col_A, *Col_B);
  } else {
    LOG("[gemm] This gemm operation is sized correctly to be compatible with our accelerator, calling that!\n");
    gemm_fpga_kern(A_re, A_im, B_re, B_im, C_re, C_im, resource_idx);
  }

  for (size_t i = 0; i < (*Row_A) * (*Col_B); i++){
    (*C)[i].re = (dash_re_flt_type) C_re[i];
    (*C)[i].im = (dash_re_flt_type) C_im[i];
  }

  free(A_re);
  free(A_im);
  free(B_re);
  free(B_im);
  free(C_re);
  free(C_im);
}
extern "C" void DASH_GEMM_int_gemm(dash_cmplx_int_type** A, dash_cmplx_int_type** B, dash_cmplx_int_type** C, size_t* Row_A, size_t* Col_A, size_t* Col_B, uint8_t resource_idx) {
  // TODO: if we change the accelerator to support interleaved Re/Im values, then we can potentially optimize this to require no copying
  LOG("[gemm] Calling DASH_GEMM_int_gemm\n");
  gemm_re_type *A_re = (gemm_re_type *) calloc((*Row_A) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *A_im = (gemm_re_type *) calloc((*Row_A) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *B_re = (gemm_re_type *) calloc((*Col_B) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *B_im = (gemm_re_type *) calloc((*Col_B) * (*Col_A), sizeof(gemm_re_type));
  gemm_re_type *C_re = (gemm_re_type *) calloc((*Row_A) * (*Col_B), sizeof(gemm_re_type));
  gemm_re_type *C_im = (gemm_re_type *) calloc((*Row_A) * (*Col_B), sizeof(gemm_re_type));

  for (size_t i = 0; i < (*Row_A) * (*Col_A); i++) {
    A_re[i] = (gemm_re_type) (*A)[i].re;
    A_im[i] = (gemm_re_type) (*A)[i].im;
  }

  for (size_t i = 0; i < (*Col_A) * (*Col_B); i++) {
    B_re[i] = (gemm_re_type) (*B)[i].re;
    B_im[i] = (gemm_re_type) (*B)[i].im;
  }

  // If we aren't exactly multiplying (4x64 * 64x4) matrices, (i) cry and (ii) fall back to a CPU impl even if it _drastically_ confuses CEDR
  // Also: note that we are comparing the size_t typed values against their #define macro equivalents here
  if (*Row_A != A_ROWS || *Col_A != A_COLS || *Col_B != B_COLS) {
    LOG("[gemm] Falling back to CPU implementation! This gemm operation is not sized correctly to be compatible with our accelerator!\n");
    _calculate_gemm_cpu(A_re, A_im, B_re, B_im, C_re, C_im, *Row_A, *Col_A, *Col_B);
  } else {
    LOG("[gemm] This gemm operation is sized correctly to be compatible with our accelerator, calling that!\n");
    gemm_fpga_kern(A_re, A_im, B_re, B_im, C_re, C_im, resource_idx);
  }

  for (size_t i = 0; i < (*Row_A) * (*Col_B); i++){
    (*C)[i].re = (dash_re_int_type) C_re[i];
    (*C)[i].im = (dash_re_int_type) C_im[i];
  }

  free(A_re);
  free(A_im);
  free(B_re);
  free(B_im);
  free(C_re);
  free(C_im);
}

#if defined(__GEMM_ENABLE_MAIN)
/*
 * Note: You probably DON'T want to use this implementation - it's just here to serve as a means of generating expected
 * output in the main function below. Probably don't use this in your CEDR user application by hooking it in through the app's JSON
 */
extern "C" void _calculate_gemm_cpu_reference_(gemm_re_type *A, gemm_re_type *Ai, gemm_re_type *B, gemm_re_type *Bi, gemm_re_type *C, gemm_re_type *Ci) {
  gemm_re_type res1, res2, res3, res4;
  gemm_re_type term1, term2, term3, term4;
  
  struct timespec func_start;
  struct timespec func_end;

  clock_gettime(CLOCK_MONOTONIC_RAW, &func_start);

  for (int i = 0; i < A_ROWS; i++) {
    for (int j = 0; j < B_COLS; j++) {
      res1 = 0; res2 = 0; res3 = 0; res4 = 0;
      // A_COLS better equal B_ROWS or the computer gods will be mad at you >:(
      for (int k = 0; k < A_COLS; k++) {
        term1 = A[i * A_COLS + k] * B[k * B_COLS + j];
        res1 += term1;

        term2 = Ai[i * A_COLS + k] * Bi[k * B_COLS + j];
        res2 += term2;

        term3 = A[i * A_COLS + k] * Bi[k * B_COLS + j];
        res3 += term3;

        term4 = Ai[i * A_COLS + k] * B[k * B_COLS + j];
        res4 += term4;
      }
      C[i * B_COLS + j] = res1 - res2;
      Ci[i * B_COLS + j] = res3 + res4;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &func_end);

  LOG("[gemm] CPU gemm took %ld ns\n", 
      (func_end.tv_sec * SEC2NANOSEC + func_end.tv_nsec) - (func_start.tv_sec * SEC2NANOSEC + func_start.tv_nsec));
}

void _check_gemm_result_(gemm_re_type *C_actual, gemm_re_type *Ci_actual, gemm_re_type *C_expected, gemm_re_type *Ci_expected) {
  int error_count = 0;
  gemm_re_type diff, c, d;

  LOG("[gemm] Checking real output\n");
  for (int i = 0; i < C_ROWS; i++) {
    for (int j = 0; j < C_COLS; j++) {
      c = C_expected[i * C_COLS + j];
      d = C_actual[i * C_COLS + j];

      diff = std::abs(c - d) / c * 100;

      if (c != 0 && diff > 0.01) {
        LOG("[gemm] ERROR - Expected = %f, Hardware GEMM = %f, row = %d, col = %d\n", c, d, i, j);
        error_count++;
      }
    }
  }

  LOG("[gemm] Checking imaginary output\n");
  for (int i = 0; i < C_ROWS; i++) {
    for (int j = 0; j < C_COLS; j++) {
      c = Ci_expected[i * C_COLS + j];
      d = Ci_actual[i * C_COLS + j];

      diff = std::abs(c - d) / c * 100;

      if (c != 0 && diff > 0.01) {
        LOG("[ERROR] Expected = %f, Hardware GEMM = %f, row = %d, col = %d\n", c, d, i, j);
        error_count++;
      }
    }
  }

  if (error_count == 0) {
    printf("[gemm] GEMM Passed!\n");
  } else {
    printf("[gemm] GEMM Failed!\n");
  }
}

int main() {

  // Choose whether the data to compute are coming from files (specified below) or generated on the fly
  bool readFile = false;
  char A_input[] = "S_real.txt";
  char Ai_input[] = "S_imag.txt";
  char B_input[] = "Shermitian_real_S.txt";
  char Bi_input[] = "Shermitian_imag_S.txt";

  gemm_re_type *A, *Ai;
  gemm_re_type *B, *Bi;
  gemm_re_type *C_actual, *Ci_actual, *C_expected, *Ci_expected;

  A = (gemm_re_type*) calloc(A_ROWS * A_COLS, sizeof(gemm_re_type));
  Ai = (gemm_re_type*) calloc(A_ROWS * A_COLS, sizeof(gemm_re_type));

  B = (gemm_re_type*) calloc(B_ROWS * B_COLS, sizeof(gemm_re_type));
  Bi = (gemm_re_type*) calloc(B_ROWS * B_COLS, sizeof(gemm_re_type));

  C_actual = (gemm_re_type*) calloc(C_ROWS * C_COLS, sizeof(gemm_re_type));
  Ci_actual = (gemm_re_type*) calloc(C_ROWS * C_COLS, sizeof(gemm_re_type));
  C_expected = (gemm_re_type*) calloc(C_ROWS * C_COLS, sizeof(gemm_re_type));
  Ci_expected = (gemm_re_type*) calloc(C_ROWS * C_COLS, sizeof(gemm_re_type));

  if (readFile) {
    FILE *fp_real, *fp_imag;

    fp_real = fopen(A_input, "r");
    if (fp_real == NULL) {
      fprintf(stderr, "[gemm] Error! Failed to open input file %s\n", A_input);
      return -1;
    }
    fp_imag = fopen(Ai_input, "r");
    if (fp_imag == NULL) {
      fprintf(stderr, "[gemm] Error! Failed to open input file %s\n", Ai_input);
      return -1;
    }
    for (int i = 0; i < A_ROWS; i++) {
      for (int j = 0; j < A_COLS; j++) {
        fscanf(fp_real, "%f", &A[i * A_COLS + j]);
        fscanf(fp_imag, "%f", &Ai[i * A_COLS + j]);
      }
    }
    fclose(fp_real); fclose(fp_imag);

    fp_real = fopen(B_input, "r");
    if (fp_real == NULL) {
      fprintf(stderr, "[gemm] Error! Failed to open input file %s\n", B_input);
      return -1;
    }
    fp_imag = fopen(Bi_input, "r");
    if (fp_imag == NULL) {
      fprintf(stderr, "[gemm] Error! Failed to open input file %s\n", Bi_input);
      return -1;
    }
    for (int i = 0; i < B_ROWS; i++) {
      for (int j = 0; j < B_COLS; j++) {
        fscanf(fp_real, "%f", &B[i * B_COLS + j]);
        fscanf(fp_imag, "%f", &Bi[i * B_COLS + j]);
      }
    }
    fclose(fp_real); fclose(fp_imag);
    LOG("[gemm] Initialized all inputs from the provided files!\n");
  } else {
    for (int i = 0; i < A_ROWS; i++) {
      for (int j = 0; j < A_COLS; j++) {
        A[i * A_COLS + j] = (gemm_re_type) (i * A_COLS + j);
        Ai[i * A_COLS + j] = (gemm_re_type) (i * A_COLS + j);
      }
    }
    for (int i = 0; i < B_ROWS; i++) {
      for (int j = 0; j < B_COLS; j++) {
        B[i * B_COLS + j] = (gemm_re_type) (i * B_COLS + j);
        Bi[i * B_COLS + j] = (gemm_re_type) (i * B_COLS + j);
      }
    }
    LOG("[gemm] Initialized all inputs with hard-coded matrices!\n");
  }

  _calculate_gemm_cpu_reference_(A, Ai, B, Bi, C_expected, Ci_expected);

  for (uint8_t gemm_num = 0; gemm_num < NUM_GEMMS; gemm_num++) {
    LOG("[gemm] Testing GEMM %u\n", gemm_num);
    gemm_fpga_kern(A, Ai, B, Bi, C_actual, Ci_actual, gemm_num);
    _check_gemm_result_(C_actual, Ci_actual, C_expected, Ci_expected);  
  }

  free(A); free(Ai); free(B); free(Bi);
  free(C_actual); free(Ci_actual); free(C_expected); free(Ci_expected);
}
#endif
