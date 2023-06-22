#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[7]) {

  int classes[2];

  if (features[0] <= 12717.0) {
    classes[0] = 53;
    classes[1] = 0;
  } else {
    if (features[0] <= 17520.5) {
      if (features[1] <= 17335.5) {
        classes[0] = 0;
        classes[1] = 35;
      } else {
        if (features[5] <= 6.5) {
          if (features[5] <= 4.5) {
            classes[0] = 0;
            classes[1] = 1;
          } else {
            classes[0] = 1;
            classes[1] = 0;
          }
        } else {
          classes[0] = 0;
          classes[1] = 3;
        }
      }
    } else {
      if (features[0] <= 30462.5) {
        classes[0] = 46;
        classes[1] = 0;
      } else {
        if (features[1] <= 47459.0) {
          if (features[1] <= 35783.5) {
            if (features[1] <= 31729.5) {
              classes[0] = 0;
              classes[1] = 13;
            } else {
              classes[0] = 18;
              classes[1] = 0;
            }
          } else {
            classes[0] = 0;
            classes[1] = 57;
          }
        } else {
          if (features[1] <= 58971.5) {
            if (features[1] <= 53376.5) {
              if (features[5] <= 6.0) {
                classes[0] = 0;
                classes[1] = 2;
              } else {
                classes[0] = 8;
                classes[1] = 0;
              }
            } else {
              classes[0] = 37;
              classes[1] = 0;
            }
          } else {
            if (features[0] <= 133933.5) {
              if (features[0] <= 129130.0) {
                if (features[1] <= 122631.5) {
                  if (features[0] <= 106460.0) {
                    if (features[1] <= 101016.5) {
                      if (features[0] <= 95128.0) {
                        if (features[1] <= 88413.0) {
                          classes[0] = 51;
                          classes[1] = 47;
                        } else {
                          classes[0] = 26;
                          classes[1] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 32;
                      }
                    } else {
                      classes[0] = 37;
                      classes[1] = 0;
                    }
                  } else {
                    if (features[1] <= 112590.0) {
                      classes[0] = 0;
                      classes[1] = 26;
                    } else {
                      if (features[0] <= 117737.5) {
                        classes[0] = 10;
                        classes[1] = 0;
                      } else {
                        classes[0] = 0;
                        classes[1] = 16;
                      }
                    }
                  }
                } else {
                  classes[0] = 30;
                  classes[1] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 44;
              }
            } else {
              if (features[0] <= 140964.0) {
                classes[0] = 24;
                classes[1] = 0;
              } else {
                classes[0] = 0;
                classes[1] = 2;
              }
            }
          }
        }
      }
    }
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