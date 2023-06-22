#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[8]) {

  int classes[3];

  if (features[0] <= 62209.0) {
    if (features[6] <= 2.5) {
      if (features[0] <= 21626.5) {
        classes[0] = 17;
        classes[1] = 0;
        classes[2] = 0;
      } else {
        if (features[0] <= 21649.5) {
          if (features[7] <= 0.5) {
            classes[0] = 0;
            classes[1] = 2;
            classes[2] = 0;
          } else {
            classes[0] = 1;
            classes[1] = 0;
            classes[2] = 0;
          }
        } else {
          if (features[6] <= 1.5) {
            classes[0] = 15;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            if (features[1] <= 41365.5) {
              classes[0] = 3;
              classes[1] = 0;
              classes[2] = 0;
            } else {
              if (features[1] <= 49848.5) {
                classes[0] = 0;
                classes[1] = 1;
                classes[2] = 0;
              } else {
                classes[0] = 1;
                classes[1] = 0;
                classes[2] = 0;
              }
            }
          }
        }
      }
    } else {
      if (features[2] <= 18694.5) {
        if (features[1] <= 18566.5) {
          if (features[0] <= 21398.0) {
            classes[0] = 49;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            if (features[0] <= 21546.5) {
              classes[0] = 0;
              classes[1] = 2;
              classes[2] = 0;
            } else {
              classes[0] = 8;
              classes[1] = 0;
              classes[2] = 0;
            }
          }
        } else {
          classes[0] = 0;
          classes[1] = 0;
          classes[2] = 1;
        }
      } else {
        classes[0] = 134;
        classes[1] = 0;
        classes[2] = 0;
      }
    }
  } else {
    if (features[6] <= 17.5) {
      if (features[0] <= 63597.5) {
        if (features[6] <= 1.5) {
          if (features[1] <= 60559.0) {
            if (features[2] <= 54393.0) {
              classes[0] = 2;
              classes[1] = 0;
              classes[2] = 0;
            } else {
              classes[0] = 0;
              classes[1] = 1;
              classes[2] = 0;
            }
          } else {
            classes[0] = 3;
            classes[1] = 0;
            classes[2] = 0;
          }
        } else {
          if (features[0] <= 62391.0) {
            classes[0] = 1;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 2;
          }
        }
      } else {
        if (features[6] <= 0.5) {
          if (features[0] <= 96865.5) {
            classes[0] = 3;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            classes[0] = 0;
            classes[1] = 1;
            classes[2] = 0;
          }
        } else {
          if (features[6] <= 4.5) {
            if (features[0] <= 74677.0) {
              if (features[6] <= 3.5) {
                classes[0] = 4;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 1;
              }
            } else {
              if (features[1] <= 99505.0) {
                classes[0] = 13;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                if (features[2] <= 88170.5) {
                  if (features[6] <= 1.5) {
                    classes[0] = 1;
                    classes[1] = 0;
                    classes[2] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                  }
                } else {
                  classes[0] = 5;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              }
            }
          } else {
            classes[0] = 50;
            classes[1] = 0;
            classes[2] = 0;
          }
        }
      }
    } else {
      if (features[6] <= 20.5) {
        if (features[7] <= 0.5) {
          if (features[0] <= 121207.0) {
            if (features[1] <= 105117.5) {
              if (features[0] <= 79907.5) {
                if (features[2] <= 55124.0) {
                  classes[0] = 0;
                  classes[1] = 1;
                  classes[2] = 0;
                } else {
                  if (features[6] <= 18.5) {
                    if (features[0] <= 74063.5) {
                      classes[0] = 1;
                      classes[1] = 0;
                      classes[2] = 0;
                    } else {
                      classes[0] = 0;
                      classes[1] = 1;
                      classes[2] = 0;
                    }
                  } else {
                    classes[0] = 3;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                }
              } else {
                classes[0] = 0;
                classes[1] = 3;
                classes[2] = 0;
              }
            } else {
              classes[0] = 2;
              classes[1] = 0;
              classes[2] = 0;
            }
          } else {
            classes[0] = 0;
            classes[1] = 2;
            classes[2] = 0;
          }
        } else {
          classes[0] = 4;
          classes[1] = 0;
          classes[2] = 0;
        }
      } else {
        classes[0] = 27;
        classes[1] = 0;
        classes[2] = 0;
      }
    }
  }

  int index = 0;
  for (int i = 0; i < 3; i++) {
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