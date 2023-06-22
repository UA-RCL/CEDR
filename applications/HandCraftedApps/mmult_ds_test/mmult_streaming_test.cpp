#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#define A_ROWS 4
#define A_COLS 64

#define B_ROWS (A_COLS)
#define B_COLS 4

#define C_ROWS (A_ROWS)
#define C_COLS (B_COLS)

float *A_compute, *Ai_compute;
float *B_compute, *Bi_compute;
float *_A_compute, *_Ai_compute;
float *_B_compute, *_Bi_compute;

float *C_print, *Ci_print;
float *_C_print, *_Ci_print;

void *dlhandle;
void (*mmult_accel_func)(int *, int *, float *, float *, float *, float *, int *, float **, float **, float *, float **,
                         float **, float *, float *, float *, float **, float **);

void __attribute__((constructor)) setup(void) {
  A_compute = (float*) calloc(A_ROWS * A_COLS, sizeof(float));
  Ai_compute = (float*) calloc(A_ROWS * A_COLS, sizeof(float));
  B_compute = (float*) calloc(B_ROWS * B_COLS, sizeof(float));
  Bi_compute = (float*) calloc(B_ROWS * B_COLS, sizeof(float));

  _A_compute = (float*) calloc(A_ROWS * A_COLS, sizeof(float));
  _Ai_compute = (float*) calloc(A_ROWS * A_COLS, sizeof(float));
  _B_compute = (float*) calloc(B_ROWS * B_COLS, sizeof(float));
  _Bi_compute = (float*) calloc(B_ROWS * B_COLS, sizeof(float));

  C_print = (float*) calloc(C_ROWS * C_COLS, sizeof(float));
  Ci_print = (float*) calloc(C_ROWS * C_COLS, sizeof(float));
  _C_print = (float*) calloc(C_ROWS * C_COLS, sizeof(float));
  _Ci_print = (float*) calloc(C_ROWS * C_COLS, sizeof(float));

  dlhandle = dlopen("./apps/mmult-aarch64.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open MMULT shared object!\n");
  }
  mmult_accel_func =
      (void (*)(int *, int *, float *, float *, float *, float *, int *, float **, float **, float *, float **,
                float **, float *, float *, float *, float **, float **))dlsym(dlhandle, "mmult_fpga_kern");
  if (mmult_accel_func == nullptr) {
    printf("Unable to get function handle for MMULT accelerator function!\n");
  }
}

void __attribute__((destructor)) teardown(void) {
  free(A_compute); free(Ai_compute);
  free(_A_compute); free(_Ai_compute);
  free(B_compute); free(Bi_compute);
  free(_B_compute); free(_Bi_compute);

  free(C_print); free(Ci_print);
  free(_C_print); free(_Ci_print);
}

extern "C" void generate_input(void) {
  static bool buffer_select = true;

  printf("About to generate mmult input, buffer_select = %d\n", buffer_select);

  if (buffer_select) {
    for (int i = 0; i < A_ROWS; i++) {
      for (int j = 0; j < A_COLS; j++) {
        A_compute[i * A_COLS + j] = 2.0f;
        Ai_compute[i * A_COLS + j] = 2.0f;
      }
    }
    for (int i = 0; i < B_ROWS; i++) {
      for (int j = 0; j < B_COLS; j++) {
        B_compute[i * B_COLS + j] = 3.0f;
        Bi_compute[i * B_COLS + j] = 3.0f;
      }
    }
  } else {
    for (int i = 0; i < A_ROWS; i++) {
      for (int j = 0; j < A_COLS; j++) {
        _A_compute[i * A_COLS + j] = 2.0f;
        _Ai_compute[i * A_COLS + j] = 2.0f;
      }
    }
    for (int i = 0; i < B_ROWS; i++) {
      for (int j = 0; j < B_COLS; j++) {
        _B_compute[i * B_COLS + j] = 3.0f;
        _Bi_compute[i * B_COLS + j] = 3.0f;
      }
    }
  }

  buffer_select = !buffer_select;
}

extern "C" void compute_mmult(void) {
  static bool buffer_select = true;


  if (buffer_select) {
    printf("About to compute mmult, buffer_select = true\n");
    mmult_accel_func(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    &A_compute, &B_compute, nullptr, &Ai_compute, &Bi_compute, nullptr, nullptr, nullptr,
                    &C_print, &Ci_print);
  } else {
    printf("About to compute mmult, buffer_select = false\n");
    mmult_accel_func(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    &_A_compute, &_B_compute, nullptr, &_Ai_compute, &_Bi_compute, nullptr, nullptr, nullptr,
                    &_C_print, &_Ci_print);
  }

  buffer_select = !buffer_select;
}

extern "C" void check_output(void) {
  static bool buffer_select = true;

  printf("Printing output matrix:\n");
  if (buffer_select) {
    printf("About to print result of mmult, buffer_select = true\n");
    for (int i = 0; i < C_ROWS; i++) {
      for (int j = 0; j < C_COLS; j++) {
        printf("(%f, %f)\t", C_print[i * C_COLS + j], Ci_print[i * C_COLS + j]);
      }
      printf("\n");
    }
  } else {
    printf("About to print result of mmult, buffer_select = false\n");
    for (int i = 0; i < C_ROWS; i++) {
      for (int j = 0; j < C_COLS; j++) {
        printf("(%f, %f)\t", _C_print[i * C_COLS + j], _Ci_print[i * C_COLS + j]);
      }
      printf("\n");
    }
  }
  printf("Finished printing output\n");

  if (buffer_select) {
    for (int i = 0; i < C_ROWS; i++) {
      for (int j = 0; j < C_COLS; j++) {
        if (std::abs(C_print[i * C_COLS + j]) > 0.001 || std::abs(Ci_print[i * C_COLS + j] - 768.0f) > 0.001) {
          printf("Error at (row, col) = (%d, %d)\n", i, j);
        }
      }
    }
  } else {
    for (int i = 0; i < C_ROWS; i++) {
      for (int j = 0; j < C_COLS; j++) {
        if (std::abs(_C_print[i * C_COLS + j]) > 0.001 || std::abs(_Ci_print[i * C_COLS + j] - 768.0f) > 0.001) {
          printf("Error at (row, col) = (%d, %d)\n", i, j);
        }
      }
    }
  }

  printf("Exiting check_output\n");

  buffer_select = !buffer_select;
}
