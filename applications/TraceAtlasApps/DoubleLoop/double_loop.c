#include <stdio.h>

int main(void) {
  int i, j, k;
  double l, m, n;

  for (i = 0; i < 10; i++) {
    l = 1.0f;
    for (j = 0; j < 1000; j++) {
      m = l * 2.0f + m;   
    }
    for (k = 0; k < 10; k++) {
      n = m / 1.234f;
    }
  }

  for (i = 0; i < 10; i++) {
    l = 1.0f;
    for (j = 0; j < 1000; j++) {
      m = l * 2.0f + m;   
    }
    for (k = 0; k < 10; k++) {
      n = m / 1.234f;
    }
  }

  printf("l: %f, m: %f, n: %f\n", l, m, n);
}
