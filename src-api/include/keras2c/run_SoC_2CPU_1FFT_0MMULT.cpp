#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[8]) {

  int classes[3];

  if (features[0] <= 109083.0) {
    if (features[0] <= 12717.0) {
      classes[0] = 56;
      classes[1] = 0;
      classes[2] = 0;
    } else {
      if (features[1] <= 47459.0) {
        if (features[0] <= 43351.0) {
          if (features[0] <= 17520.5) {
            if (features[6] <= 0.5) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 1;
            } else {
              if (features[1] <= 17330.5) {
                classes[0] = 0;
                classes[1] = 28;
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
            }
          } else {
            if (features[0] <= 30461.5) {
              classes[0] = 43;
              classes[1] = 0;
              classes[2] = 0;
            } else {
              if (features[1] <= 31729.5) {
                classes[0] = 0;
                classes[1] = 17;
                classes[2] = 0;
              } else {
                if (features[6] <= 0.5) {
                  if (features[2] <= 32012.0) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 2;
                  } else {
                    classes[0] = 1;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                } else {
                  classes[0] = 19;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              }
            }
          }
        } else {
          classes[0] = 0;
          classes[1] = 57;
          classes[2] = 0;
        }
      } else {
        if (features[1] <= 88387.0) {
          if (features[0] <= 83822.0) {
            if (features[0] <= 76117.0) {
              if (features[0] <= 71319.5) {
                if (features[0] <= 58797.0) {
                  if (features[6] <= 5.5) {
                    if (features[1] <= 53376.5) {
                      if (features[0] <= 48586.5) {
                        if (features[1] <= 48573.5) {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 0;
                        } else {
                          classes[0] = 2;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 3;
                        classes[2] = 0;
                      }
                    } else {
                      if (features[6] <= 2.5) {
                        if (features[0] <= 52490.0) {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 2;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 4;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    }
                  } else {
                    classes[0] = 31;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                } else {
                  if (features[1] <= 64650.5) {
                    if (features[6] <= 3.0) {
                      if (features[6] <= 1.5) {
                        classes[0] = 0;
                        classes[1] = 2;
                        classes[2] = 0;
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                      }
                    } else {
                      classes[0] = 0;
                      classes[1] = 14;
                      classes[2] = 0;
                    }
                  } else {
                    classes[0] = 22;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                }
              } else {
                if (features[1] <= 72116.0) {
                  classes[0] = 0;
                  classes[1] = 8;
                  classes[2] = 0;
                } else {
                  if (features[6] <= 10.5) {
                    classes[0] = 0;
                    classes[1] = 1;
                    classes[2] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 2;
                  }
                }
              }
            } else {
              if (features[6] <= 17.5) {
                classes[0] = 22;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                if (features[2] <= 72602.5) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 1;
                } else {
                  classes[0] = 7;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              }
            }
          } else {
            classes[0] = 0;
            classes[1] = 19;
            classes[2] = 0;
          }
        } else {
          if (features[0] <= 97804.0) {
            if (features[6] <= 0.5) {
              if (features[0] <= 91889.0) {
                classes[0] = 2;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 2;
              }
            } else {
              if (features[0] <= 90761.0) {
                if (features[0] <= 90543.0) {
                  classes[0] = 28;
                  classes[1] = 0;
                  classes[2] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 2;
                }
              } else {
                classes[0] = 37;
                classes[1] = 0;
                classes[2] = 0;
              }
            }
          } else {
            if (features[1] <= 100168.0) {
              classes[0] = 0;
              classes[1] = 6;
              classes[2] = 0;
            } else {
              if (features[6] <= 17.5) {
                classes[0] = 12;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                if (features[6] <= 21.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 2;
                } else {
                  classes[0] = 3;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if (features[1] <= 124497.5) {
      if (features[1] <= 111930.5) {
        classes[0] = 0;
        classes[1] = 28;
        classes[2] = 0;
      } else {
        if (features[0] <= 120339.0) {
          classes[0] = 7;
          classes[1] = 0;
          classes[2] = 0;
        } else {
          if (features[6] <= 19.5) {
            if (features[6] <= 2.5) {
              if (features[2] <= 119207.5) {
                if (features[2] <= 118167.5) {
                  classes[0] = 0;
                  classes[1] = 2;
                  classes[2] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 1;
                }
              } else {
                classes[0] = 0;
                classes[1] = 3;
                classes[2] = 0;
              }
            } else {
              classes[0] = 0;
              classes[1] = 20;
              classes[2] = 0;
            }
          } else {
            if (features[1] <= 117136.5) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 1;
            } else {
              classes[0] = 0;
              classes[1] = 4;
              classes[2] = 0;
            }
          }
        }
      }
    } else {
      if (features[0] <= 131746.0) {
        if (features[6] <= 17.5) {
          classes[0] = 15;
          classes[1] = 0;
          classes[2] = 0;
        } else {
          if (features[0] <= 125571.0) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 1;
          } else {
            classes[0] = 3;
            classes[1] = 0;
            classes[2] = 0;
          }
        }
      } else {
        if (features[6] <= 18.5) {
          classes[0] = 0;
          classes[1] = 11;
          classes[2] = 0;
        } else {
          if (features[6] <= 20.5) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 1;
          } else {
            classes[0] = 0;
            classes[1] = 4;
            classes[2] = 0;
          }
        }
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