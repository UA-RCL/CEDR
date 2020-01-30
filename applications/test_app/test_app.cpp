#include <cstdio>
#include "test_app.hpp"

extern "C" void test_app_node0(int* x, int* y) {
    *x = *x + 5;
    *y = *y + 2;
}

extern "C" void test_app_node1(double* w, double** dbl_array) {
    *w = *w + 12.34f;
    for (int i = 0; i < 10; i++) {
      (*dbl_array)[i] = 1.1f * i;
    }
}

extern "C" void test_app_node2(int* x, int* y, int* z) {
    *z = *y * *x;
}

extern "C" void test_app_node3_platform1(double* w, int* z) {
    printf("I am node 3 platform 1. I'm running on the CPU\n");
    *w = *w + 1.1f + *z;
}

extern "C" void test_app_node3_platform2(double* w, int* z) {
    printf("I am node 3 platform 2. I'm supposedly dispatching onto an accelerator\n");
    *w = *w + 1.1f + *z;
}

extern "C" void test_app_node4(double* w, int* x, int* y, int* z, double** dbl_array) {
    printf("test_app finished execution. ");
    printf("output: {w: %f, x: %d, y: %d, z: %d\n", *w, *x, *y, *z);
    printf("\tdbl_array: ");
    for (int i = 0; i < 10; i++) {
      if (i < 9) {
        printf("%f, ", (*dbl_array)[i]);
      } else {
        printf("%f", (*dbl_array)[i]);
      }
    }
    printf("}\n");
}

int main(void) {}