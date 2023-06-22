#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[8]) {

  int classes[3];

  if (features[0] <= 12717.0) {
    classes[0] = 49;
    classes[1] = 0;
    classes[2] = 0;
  } else {
    if (features[0] <= 128812.0) {
      if (features[0] <= 17520.5) {
        if (features[1] <= 17335.5) {
          classes[0] = 0;
          classes[1] = 30;
          classes[2] = 0;
        } else {
          if (features[6] <= 6.5) {
            if (features[6] <= 4.5) {
              classes[0] = 0;
              classes[1] = 1;
              classes[2] = 0;
            } else {
              classes[0] = 1;
              classes[1] = 0;
              classes[2] = 0;
            }
          } else {
            classes[0] = 0;
            classes[1] = 3;
            classes[2] = 0;
          }
        }
      } else {
        if (features[0] <= 30461.5) {
          classes[0] = 47;
          classes[1] = 0;
          classes[2] = 0;
        } else {
          if (features[0] <= 123405.0) {
            if (features[0] <= 48154.5) {
              if (features[1] <= 35784.0) {
                if (features[1] <= 31729.5) {
                  classes[0] = 0;
                  classes[1] = 13;
                  classes[2] = 0;
                } else {
                  classes[0] = 17;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 51;
                classes[2] = 0;
              }
            } else {
              if (features[0] <= 58797.0) {
                if (features[1] <= 53376.5) {
                  if (features[0] <= 48586.5) {
                    if (features[6] <= 0.5) {
                      if (features[0] <= 48567.5) {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                      } else {
                        classes[0] = 1;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 6;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  } else {
                    classes[0] = 0;
                    classes[1] = 3;
                    classes[2] = 0;
                  }
                } else {
                  classes[0] = 41;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              } else {
                if (features[1] <= 103783.5) {
                  if (features[0] <= 96331.0) {
                    if (features[1] <= 64654.0) {
                      classes[0] = 0;
                      classes[1] = 20;
                      classes[2] = 0;
                    } else {
                      if (features[0] <= 89213.5) {
                        if (features[0] <= 83807.0) {
                          classes[0] = 40;
                          classes[1] = 13;
                          classes[2] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 18;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 36;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    }
                  } else {
                    if (features[1] <= 103500.0) {
                      if (features[6] <= 37.5) {
                        if (features[6] <= 2.5) {
                          classes[0] = 0;
                          classes[1] = 11;
                          classes[2] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 37;
                          classes[2] = 0;
                        }
                      } else {
                        if (features[0] <= 101721.5) {
                          classes[0] = 0;
                          classes[1] = 3;
                          classes[2] = 0;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      }
                    } else {
                      if (features[1] <= 103545.5) {
                        classes[0] = 2;
                        classes[1] = 0;
                        classes[2] = 0;
                      } else {
                        if (features[6] <= 14.5) {
                          classes[0] = 0;
                          classes[1] = 7;
                          classes[2] = 0;
                        } else {
                          classes[0] = 2;
                          classes[1] = 1;
                          classes[2] = 0;
                        }
                      }
                    }
                  }
                } else {
                  if (features[0] <= 113364.0) {
                    if (features[1] <= 108733.0) {
                      if (features[1] <= 103944.5) {
                        if (features[6] <= 13.5) {
                          classes[0] = 10;
                          classes[1] = 0;
                          classes[2] = 0;
                        } else {
                          classes[0] = 5;
                          classes[1] = 2;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 2;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 24;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  } else {
                    if (features[1] <= 119973.0) {
                      classes[0] = 0;
                      classes[1] = 11;
                      classes[2] = 0;
                    } else {
                      classes[0] = 11;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  }
                }
              }
            }
          } else {
            classes[0] = 0;
            classes[1] = 29;
            classes[2] = 0;
          }
        }
      }
    } else {
      if (features[2] <= 110758.5) {
        if (features[0] <= 129827.5) {
          classes[0] = 7;
          classes[1] = 0;
          classes[2] = 0;
        } else {
          classes[0] = 0;
          classes[1] = 0;
          classes[2] = 1;
        }
      } else {
        classes[0] = 34;
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