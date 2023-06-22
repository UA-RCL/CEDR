#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static bool fft_buffer_select = true;

double *fft_input_ref, *fft_output_ref;

double *fft_input_compute, *_fft_input_compute;
double *fft_output_compute, *_fft_output_compute;
double *fft_output_print, *_fft_output_print;

void *dlhandle;
void (*fft_cpu_func)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**);
void (*fft_accel_func)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**);

void __attribute__((constructor)) setup(void) {

  fft_input_ref = (double*) calloc(2 * 256, sizeof(double));
  fft_output_ref = (double*) calloc(2 * 256, sizeof(double));

  fft_input_ref[0] = 1;
  for (size_t i = 0; i < 2 * 256; i++) {
    if (i % 2 == 0) {
      fft_output_ref[i] = 1;
    } else {
      fft_output_ref[i] = 0;
    }
  }

  fft_input_compute = (double*) calloc(2 * 256, sizeof(double));
  _fft_input_compute = (double*) calloc(2 * 256, sizeof(double));

  fft_output_compute = (double*) calloc(2 * 256, sizeof(double));
  _fft_output_compute = (double*) calloc(2 * 256, sizeof(double));

  fft_output_print = (double*) calloc(2 * 256, sizeof(double));
  _fft_output_print = (double*) calloc(2 * 256, sizeof(double));

  dlhandle = dlopen("./apps/fft-aarch64.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open FFT shared object!\n");
  }
  fft_accel_func =
      (void (*)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**))dlsym(dlhandle, "fft256_accel");
  if (fft_accel_func == nullptr) {
    printf("Unable to get function handle for FFT accelerator function!\n");
  }
  fft_cpu_func =
      (void (*)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**))dlsym(dlhandle, "fft256_cpu");
  if (fft_cpu_func == nullptr) {
    printf("Unable to get function handle for FFT CPU function!\n");
  }
}

void __attribute__((destructor)) teardown(void) {
  free(fft_input_ref);
  free(fft_output_ref);

  free(fft_input_compute);
  free(_fft_input_compute);

  free(fft_output_compute);
  free(_fft_output_compute);

  free(fft_output_print);
  free(_fft_output_print);
}

extern "C" void generate_input(void) {
  static bool buffer_select = true;

  if (buffer_select) {
    memcpy(fft_input_compute, fft_input_ref, 2 * 256 * sizeof(double));
  } else {
    memcpy(_fft_input_compute, fft_input_ref, 2 * 256 * sizeof(double));
  }

  buffer_select = !buffer_select;
}

extern "C" void compute_fft_cpu(void) {
  if (fft_buffer_select) {
    fft_cpu_func(nullptr, nullptr, nullptr, nullptr, nullptr, &fft_input_compute, &fft_output_compute);
  } else {
    fft_cpu_func(nullptr, nullptr, nullptr, nullptr, nullptr, &_fft_input_compute, &_fft_output_compute);
  }

  fft_buffer_select = !fft_buffer_select;
}

extern "C" void compute_fft_accel(void) {
  if (fft_buffer_select) {
    fft_accel_func(nullptr, nullptr, nullptr, nullptr, nullptr, &fft_input_compute, &fft_output_compute);
  } else {
    fft_accel_func(nullptr, nullptr, nullptr, nullptr, nullptr, &_fft_input_compute, &_fft_output_compute);
  }

  fft_buffer_select = !fft_buffer_select;
}

extern "C" void check_output(void) {
  static bool buffer_select = true;

  int error_count = 0;
  float diff, c, d;

  printf("[FFT Streaming] Checking actual versus expected output\n");
  for (size_t i = 0; i < 2 * 256; i++) {
    c = fft_output_ref[i];
    if (buffer_select) {
      d = fft_output_compute[i];
    } else {
      d = _fft_output_compute[i];
    }

    diff = std::abs(c - d) / c * 100;

    if (diff > 0.01) {
      fprintf(stderr, "[FFT Streaming] [ERROR] Expected = %f, Actual FFT = %f, index = %ld\n", c, d, i);
      error_count++;
    }
  }

  if (error_count == 0) {
    printf("[FFT Streaming] FFT Passed!\n");
  } else {
    fprintf(stderr, "[FFT Streaming] FFT Failed!\n");
  }

  buffer_select = !buffer_select;
}
