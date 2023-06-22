#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[7]) {

  int classes[2];

  if (features[5] <= 4.5) {
    if (features[5] <= 3.5) {
      if (features[0] <= 145252.5) {
        if (features[1] <= 61380.5) {
          classes[0] = 62;
          classes[1] = 0;
        } else {
          if (features[0] <= 63654.5) {
            if (features[5] <= 1.0) {
              classes[0] = 1;
              classes[1] = 0;
            } else {
              classes[0] = 0;
              classes[1] = 1;
            }
          } else {
            classes[0] = 40;
            classes[1] = 0;
          }
        }
      } else {
        if (features[1] <= 136041.0) {
          if (features[5] <= 1.5) {
            classes[0] = 2;
            classes[1] = 0;
          } else {
            classes[0] = 0;
            classes[1] = 1;
          }
        } else {
          classes[0] = 11;
          classes[1] = 0;
        }
      }
    } else {
      if (features[0] <= 145256.0) {
        if (features[1] <= 54406.0) {
          classes[0] = 15;
          classes[1] = 0;
        } else {
          if (features[1] <= 60714.5) {
            classes[0] = 0;
            classes[1] = 1;
          } else {
            if (features[1] <= 128508.5) {
              if (features[0] <= 132583.0) {
                if (features[1] <= 71027.5) {
                  if (features[0] <= 69076.5) {
                    classes[0] = 2;
                    classes[1] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 1;
                  }
                } else {
                  if (features[0] <= 115409.0) {
                    classes[0] = 6;
                    classes[1] = 0;
                  } else {
                    if (features[0] <= 121054.5) {
                      classes[0] = 0;
                      classes[1] = 1;
                    } else {
                      classes[0] = 4;
                      classes[1] = 0;
                    }
                  }
                }
              } else {
                classes[0] = 0;
                classes[1] = 1;
              }
            } else {
              classes[0] = 5;
              classes[1] = 0;
            }
          }
        }
      } else {
        if (features[1] <= 144240.5) {
          classes[0] = 0;
          classes[1] = 2;
        } else {
          classes[0] = 2;
          classes[1] = 0;
        }
      }
    }
  } else {
    classes[0] = 325;
    classes[1] = 0;
  }

  int index = 0;
  for (int i = 0; i < 2; i++) {
    index = classes[i] > classes[index] ? i : index;
  }
  return index;
}

// int main(int argc, const char * argv[]) {
//
//    /* Features: */
//    double features[argc-1];
//    int i;
//    for (i = 1; i < argc; i++) {
//        features[i-1] = atof(argv[i]);
//    }
//
//    /* Prediction: */
//    printf("%d", predict(features));
//    return 0;
//
//}