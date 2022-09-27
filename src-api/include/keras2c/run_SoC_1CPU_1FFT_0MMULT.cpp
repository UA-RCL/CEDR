#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[7]) {

  int classes[2];

  if (features[5] <= 0.5) {
    if (features[0] <= 21593.0) {
      classes[0] = 8;
      classes[1] = 0;
    } else {
      if (features[0] <= 41322.0) {
        if (features[0] <= 29264.5) {
          if (features[0] <= 23658.0) {
            classes[0] = 0;
            classes[1] = 1;
          } else {
            classes[0] = 1;
            classes[1] = 0;
          }
        } else {
          classes[0] = 0;
          classes[1] = 2;
        }
      } else {
        if (features[1] <= 98879.5) {
          if (features[1] <= 71847.5) {
            if (features[0] <= 69251.5) {
              if (features[0] <= 49876.0) {
                if (features[0] <= 49819.5) {
                  classes[0] = 2;
                  classes[1] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 1;
                }
              } else {
                classes[0] = 4;
                classes[1] = 0;
              }
            } else {
              classes[0] = 0;
              classes[1] = 1;
            }
          } else {
            classes[0] = 5;
            classes[1] = 0;
          }
        } else {
          if (features[0] <= 132503.5) {
            classes[0] = 0;
            classes[1] = 3;
          } else {
            if (features[0] <= 138840.0) {
              classes[0] = 2;
              classes[1] = 0;
            } else {
              if (features[0] <= 145099.5) {
                classes[0] = 0;
                classes[1] = 1;
              } else {
                if (features[1] <= 148402.5) {
                  if (features[0] <= 145161.5) {
                    if (features[0] <= 145120.5) {
                      classes[0] = 1;
                      classes[1] = 0;
                    } else {
                      classes[0] = 0;
                      classes[1] = 1;
                    }
                  } else {
                    classes[0] = 2;
                    classes[1] = 0;
                  }
                } else {
                  if (features[1] <= 165950.0) {
                    classes[0] = 0;
                    classes[1] = 2;
                  } else {
                    classes[0] = 1;
                    classes[1] = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if (features[5] <= 2.5) {
      if (features[5] <= 1.5) {
        classes[0] = 35;
        classes[1] = 0;
      } else {
        if (features[0] <= 121262.0) {
          if (features[1] <= 21649.5) {
            if (features[0] <= 14442.0) {
              classes[0] = 4;
              classes[1] = 0;
            } else {
              classes[0] = 0;
              classes[1] = 2;
            }
          } else {
            if (features[0] <= 96841.5) {
              if (features[0] <= 41360.0) {
                if (features[0] <= 32886.5) {
                  classes[0] = 4;
                  classes[1] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 1;
                }
              } else {
                classes[0] = 8;
                classes[1] = 0;
              }
            } else {
              if (features[1] <= 93869.5) {
                classes[0] = 0;
                classes[1] = 1;
              } else {
                classes[0] = 3;
                classes[1] = 0;
              }
            }
          }
        } else {
          if (features[1] <= 135825.0) {
            if (features[0] <= 132578.0) {
              if (features[1] <= 127672.0) {
                classes[0] = 0;
                classes[1] = 1;
              } else {
                classes[0] = 1;
                classes[1] = 0;
              }
            } else {
              classes[0] = 0;
              classes[1] = 2;
            }
          } else {
            if (features[0] <= 174030.0) {
              if (features[1] <= 149666.0) {
                if (features[1] <= 142730.0) {
                  if (features[0] <= 145143.5) {
                    classes[0] = 2;
                    classes[1] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 1;
                  }
                } else {
                  classes[0] = 4;
                  classes[1] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 2;
              }
            } else {
              classes[0] = 4;
              classes[1] = 0;
            }
          }
        }
      }
    } else {
      if (features[5] <= 17.5) {
        classes[0] = 276;
        classes[1] = 0;
      } else {
        if (features[5] <= 20.5) {
          if (features[0] <= 79904.0) {
            if (features[1] <= 54886.0) {
              classes[0] = 13;
              classes[1] = 0;
            } else {
              if (features[1] <= 59927.5) {
                if (features[5] <= 19.5) {
                  classes[0] = 1;
                  classes[1] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 1;
                }
              } else {
                classes[0] = 4;
                classes[1] = 0;
              }
            }
          } else {
            if (features[0] <= 132530.5) {
              if (features[6] <= 0.5) {
                if (features[0] <= 109961.0) {
                  classes[0] = 0;
                  classes[1] = 3;
                } else {
                  if (features[0] <= 115599.5) {
                    classes[0] = 1;
                    classes[1] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 3;
                  }
                }
              } else {
                classes[0] = 1;
                classes[1] = 0;
              }
            } else {
              if (features[1] <= 154675.5) {
                classes[0] = 8;
                classes[1] = 0;
              } else {
                if (features[6] <= 0.5) {
                  classes[0] = 0;
                  classes[1] = 2;
                } else {
                  classes[0] = 1;
                  classes[1] = 0;
                }
              }
            }
          }
        } else {
          classes[0] = 85;
          classes[1] = 0;
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