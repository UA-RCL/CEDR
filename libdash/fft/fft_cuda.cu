#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cufft.h>

cufftDoubleComplex *d_fft256, *d_fft512, *d_fft2048;
cufftHandle plan_fft256, plan_fft512, plan_fft2048;

void __attribute__((constructor)) setup(void) {
    printf("[gpu] Allocating initial memory buffers and baseline FFT plans\n");
    
    cudaMalloc((void**) &d_fft256, sizeof(cufftDoubleComplex)*256);
    cudaMalloc((void**) &d_fft512, sizeof(cufftDoubleComplex)*512);
    cudaMalloc((void**) &d_fft2048, sizeof(cufftDoubleComplex)*2048);

    cufftPlan1d(&plan_fft256, 256, CUFFT_Z2Z, 1);
    cufftPlan1d(&plan_fft512, 512, CUFFT_Z2Z, 1);
    cufftPlan1d(&plan_fft2048, 2048, CUFFT_Z2Z, 1);

    printf("[gpu] Initial allocation complete!\n");
}
  
void __attribute__((destructor)) teardown(void) {
    printf("[gpu] Tearing down GPU memory buffers and FFT plans\n");

    cufftDestroy(plan_fft256);
    cufftDestroy(plan_fft512);
    cufftDestroy(plan_fft2048);

    // cudaFree in a destructor causes a segfault. Perhaps it's running after CUDA libraries have finished de-initializing?
    // cudaFree(d_fft256);
    // cudaFree(d_fft512);
    // cudaFree(d_fft2048);

    printf("[gpu] Teardown complete!\n");
}

void fft_cuda(double* input, double* output, int64_t fft_size, bool isForwardTransform, cufftHandle plan, cufftDoubleComplex *d_data) {
    printf("===========================================================\n");
    printf("=============== Executing %ld-Pt %s on GPU ================\n", fft_size, isForwardTransform ? "FFT" : "IFFT");
    printf("===========================================================\n");

    double *tmp_d_data = (double*) d_data;
    cudaMemcpy(tmp_d_data, input, 2 * fft_size * sizeof(input[0]), cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, d_data, d_data, isForwardTransform ? CUFFT_FORWARD : CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, tmp_d_data, 2 * fft_size * sizeof(output[0]), cudaMemcpyDeviceToHost);
}

extern "C" void fft256_cuda(int64_t *i_unused, int64_t *dftSize, int32_t *row, int32_t *col, double **dftMatrix, double **c, double **X1) {
    double *input = *c;
    double *output = *X1;
    fft_cuda(input, output, 256, true, plan_fft256, d_fft256);
}

extern "C" void ifft256_cuda(int64_t *i_unused, int64_t *dftSize, int32_t *row, int32_t *col, double **dftMatrix, double **c, double **X1) {
  double *input = *c;
  double *output = *X1;
  fft_cuda(input, output, 256, false, plan_fft256, d_fft256);
}

// extern "C" void fft256_cuda_gsl(gsl_vector_complex **input, int32_t *i, gsl_fft_complex_wavetable **wavetable, gsl_fft_complex_workspace **workspace) {
//     double *inp = (*input)->data;
//     // GSL writes the output back into the same buffer it read the data in from
//     double *out = (*input)->data;
//     fft_cuda(inp, out, 256, true);
// }

extern "C" void fft512_cuda_gsl_mex(double **input, double **output, size_t *size) {
    double *inp = (*input);
    double *out = (*output);
    fft_cuda(inp, out, 512, true, plan_fft512, d_fft512);
  }

extern "C" void fft2048_cuda(double **input, double **output, int64_t *len) {
    double *inp = *input;
    double *out = *output;
    fft_cuda(inp, out, 2048, true, plan_fft2048, d_fft2048);
}

uint32_t _check_fft_result_(double *fft_actual, double *fft_expected, size_t FFT_DIM) {
    int error_count = 0;
    float diff, c, d;
  
    printf("[fft] Checking actual versus expected output\n");
    for (size_t i = 0; i < 2 * FFT_DIM; i++) {
      c = fft_expected[i];
      d = fft_actual[i];
  
      diff = std::abs(c - d) / c * 100;
  
      if (diff > 0.01) {
        fprintf(stderr, "[fft] [ERROR] Expected = %f, Actual FFT = %f, index = %ld\n", c, d, i);
        error_count++;
      }
    }
  
    if (error_count == 0) {
      printf("[fft] FFT Passed!\n");
      return 0;
    } else {
      fprintf(stderr, "[fft] FFT Failed!\n");
      return 1;
    }
  }
  
  void _generate_fft_test_values_(double *fft_input, double *fft_expected, size_t FFT_SIZE, bool sparse_input) {
    switch (FFT_SIZE) {
    case 256:
    case 512:
    case 2048:
      if (sparse_input) {
        // Let's just do a delta function, the expected output is easy to calculate that way
        fft_input[0] = 1;
        // As such, the expected output is the constant 1 function
        for (size_t i = 0; i < FFT_SIZE; i++) {
          fft_expected[2 * i] = 1;
          fft_expected[2 * i + 1] = 0;
        }
      } else {
        // Otherwise, let's do all 1's as the input
        for (size_t i = 0; i < FFT_SIZE; i++) {
          fft_input[2 * i] = 1;
          fft_input[2 * i + 1] = 0;
        }
        // And as such, the output is some form of delta function (height = FFT_SIZE?)
        fft_expected[0] = FFT_SIZE;
      }
      break;
    default:
      fprintf(stderr, "[fft] Unknown FFT_SIZE specified in _generate_fft_test_values_: %ld\n", FFT_SIZE);
      exit(1);
    }
  }
  
  void _print_fft_result_(double *fft_result, size_t FFT_SIZE, bool isForwardTransform) {
    printf("[fft] Printing received %ld-Pt %s output\n", FFT_SIZE, isForwardTransform ? "FFT" : "IFFT");
    for (size_t i = 0; i < FFT_SIZE; i++) {
      printf("(%f + %fi)", fft_result[2 * i], fft_result[2 * i + 1]);
      if (i < FFT_SIZE - 1) {
        printf(", ");
      }
    }
    printf("\n");
  }
  
  uint32_t _run_fft_tests_(size_t FFT_SIZE, bool sparse_input, bool exit_on_failure, bool print_intermediate_results) {
    double *fft_input, *fft_expected, *fft_actual_cpu, *fft_actual_accel;
    const bool isForwardTransform = true;
    const char* FFT_TYPE = isForwardTransform ? "FFT" : "IFFT";
  
    uint32_t tests_failed = 0;

    cufftHandle plan;
    cufftDoubleComplex *d_fftData;
  
    printf("###################################################\n");
    if (sparse_input) {
      printf("[fft] Testing %ld-Pt %s implementations (sparse)\n", FFT_SIZE, FFT_TYPE);
    } else {
      printf("[fft] Testing %ld-Pt %s implementations (dense) \n", FFT_SIZE, FFT_TYPE);
    }
    printf("###################################################\n");

    if (FFT_SIZE == 256 && isForwardTransform) {
        plan = plan_fft256;
        d_fftData = d_fft256;
    } else if (FFT_SIZE == 512 && isForwardTransform) {
        plan = plan_fft512;
        d_fftData = d_fft512;
    } else if (FFT_SIZE == 2048 && isForwardTransform) {
        plan = plan_fft2048;
        d_fftData = d_fft2048;
    }
  
    fft_input = (double *)calloc(2 * FFT_SIZE, sizeof(double));
    fft_expected = (double *)calloc(2 * FFT_SIZE, sizeof(double));
    fft_actual_cpu = (double *)calloc(2 * FFT_SIZE, sizeof(double));
    fft_actual_accel = (double *)calloc(2 * FFT_SIZE, sizeof(double));
  
    printf("[fft] Generating input data and expected output\n");
    _generate_fft_test_values_(fft_input, fft_expected, FFT_SIZE, sparse_input);

    printf("[fft] Computing %ld-Pt FFT using the GPU kernel\n", FFT_SIZE);
    fft_cuda(fft_input, fft_actual_accel, FFT_SIZE, isForwardTransform, plan, d_fftData);
  
    if (print_intermediate_results) {
      _print_fft_result_(fft_actual_accel, FFT_SIZE, isForwardTransform);
    }
  
    printf("[fft] Checking the GPU output against the reference expected values\n");
    tests_failed += _check_fft_result_(fft_actual_accel, fft_expected, FFT_SIZE);
  
    if (exit_on_failure && tests_failed > 0) {
      exit(1);
    }
  
    printf("[fft] %ld-Pt FFT tests complete!\n", FFT_SIZE);
  
    free(fft_input);
    free(fft_expected);
    free(fft_actual_cpu);
    free(fft_actual_accel);
  
    return tests_failed;
  }
  
  int main() {
    // Stores the cumulative total number of tests that have failed
    uint32_t tests_failed = 0;
    // If true, exit upon the first failed test ("fail fast")
    bool exit_on_failure = true;
    // If true, print full FFT output from each implementation as it runs
    bool print_intermediate_outputs = false;
  
    // Sparse 256-Pt tests
    tests_failed += _run_fft_tests_(256, true, exit_on_failure, print_intermediate_outputs);
    // Dense 256-Pt tests
    tests_failed += _run_fft_tests_(256, false, exit_on_failure, print_intermediate_outputs);
    // Sparse 512-Pt tests
    tests_failed += _run_fft_tests_(512, true, exit_on_failure, print_intermediate_outputs);
    // Dense 512-Pt tests
    tests_failed += _run_fft_tests_(512, false, exit_on_failure, print_intermediate_outputs);
    // Sparse 2048-Pt tests
    tests_failed += _run_fft_tests_(2048, true, exit_on_failure, print_intermediate_outputs);
    // Dense 2048-Pt tests
    tests_failed += _run_fft_tests_(2048, false, exit_on_failure, print_intermediate_outputs);
  
    printf("[fft] All FFT tests completed - %d tests failed!\n", tests_failed);
    return (tests_failed > 0) ? 1 : 0;
  }