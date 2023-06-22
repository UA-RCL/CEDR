#include "../include/DashExtras.h"
#include <stdio.h>

int main(void) {
  // Note: initializing variables in the for loop causes initialization 
  // before the actual the "kernel" which makes dagExtractor attribute that store to the previous kernel 
  // (even though it actually just lies in the space between the kernels)
  int x = 0, y = 0, z = 0;
  int i = 0;

  for (; i < 1; i++) {}

  KERN_ENTER("loop");
  for (; x < 513; x++) {
    printf(".");
  }
  printf("\n");
  KERN_EXIT("loop");

  KERN_ENTER("loop");
  for (; y < 513; y++) {
    printf("-");
  }
  printf("\n");
  KERN_EXIT("loop");

  printf("Final kernel before execution ends\n");

  KERN_ENTER("loop");
  for (; z < x+y; z++) {
    printf("*");
  }
  printf("\n");
  KERN_EXIT("loop");

  printf("Exiting... (x + y = %d)\n", x+y);
  for (; i < 2; i++) {}

  return 0;
}
