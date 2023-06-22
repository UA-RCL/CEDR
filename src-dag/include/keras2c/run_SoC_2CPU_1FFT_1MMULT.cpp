#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[9]) {

  int classes[4];

  if (features[0] <= 12717.0) {
    classes[0] = 46;
    classes[1] = 0;
    classes[2] = 0;
    classes[3] = 0;
  } else {
    if (features[0] <= 17520.5) {
      if (features[7] <= 0.5) {
        if (features[1] <= 14106.0) {
          classes[0] = 0;
          classes[1] = 2;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          classes[0] = 0;
          classes[1] = 0;
          classes[2] = 1;
          classes[3] = 0;
        }
      } else {
        classes[0] = 0;
        classes[1] = 28;
        classes[2] = 0;
        classes[3] = 0;
      }
    } else {
      if (features[0] <= 30461.5) {
        classes[0] = 40;
        classes[1] = 0;
        classes[2] = 0;
        classes[3] = 0;
      } else {
        if (features[0] <= 48154.5) {
          if (features[0] <= 43351.0) {
            if (features[1] <= 31729.5) {
              classes[0] = 0;
              classes[1] = 13;
              classes[2] = 0;
              classes[3] = 0;
            } else {
              if (features[7] <= 0.5) {
                if (features[2] <= 32012.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 2;
                  classes[3] = 0;
                } else {
                  classes[0] = 1;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 0;
                }
              } else {
                classes[0] = 16;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 0;
              }
            }
          } else {
            classes[0] = 0;
            classes[1] = 54;
            classes[2] = 0;
            classes[3] = 0;
          }
        } else {
          if (features[0] <= 104007.0) {
            if (features[1] <= 87181.5) {
              if (features[0] <= 83819.0) {
                if (features[0] <= 58797.0) {
                  if (features[1] <= 53376.5) {
                    if (features[7] <= 6.5) {
                      classes[0] = 0;
                      classes[1] = 4;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      classes[0] = 8;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  } else {
                    classes[0] = 38;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 0;
                  }
                } else {
                  if (features[1] <= 64650.5) {
                    if (features[7] <= 0.5) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                      classes[3] = 0;
                    } else {
                      if (features[7] <= 3.0) {
                        if (features[1] <= 59826.0) {
                          classes[0] = 0;
                          classes[1] = 2;
                          classes[2] = 0;
                          classes[3] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 1;
                          classes[3] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 14;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    }
                  } else {
                    if (features[2] <= 64633.0) {
                      classes[0] = 16;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      if (features[1] <= 77075.0) {
                        if (features[0] <= 71319.5) {
                          classes[0] = 2;
                          classes[1] = 0;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 10;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        if (features[7] <= 2.5) {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 2;
                          classes[3] = 0;
                        } else {
                          classes[0] = 14;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    }
                  }
                }
              } else {
                if (features[7] <= 19.5) {
                  if (features[1] <= 82326.5) {
                    classes[0] = 0;
                    classes[1] = 13;
                    classes[2] = 0;
                    classes[3] = 0;
                  } else {
                    if (features[3] <= 78363.5) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 1;
                    } else {
                      classes[0] = 0;
                      classes[1] = 4;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  }
                } else {
                  if (features[7] <= 20.5) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                    classes[3] = 0;
                  } else {
                    classes[0] = 0;
                    classes[1] = 3;
                    classes[2] = 0;
                    classes[3] = 0;
                  }
                }
              }
            } else {
              if (features[7] <= 0.5) {
                if (features[0] <= 91472.5) {
                  classes[0] = 1;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  if (features[2] <= 92835.0) {
                    if (features[2] <= 89899.5) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                      classes[3] = 0;
                    } else {
                      if (features[2] <= 90744.0) {
                        classes[0] = 1;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        if (features[1] <= 91985.5) {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 2;
                          classes[3] = 0;
                        }
                      }
                    }
                  } else {
                    classes[0] = 1;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 0;
                  }
                }
              } else {
                if (features[8] <= 6.0) {
                  if (features[7] <= 2.5) {
                    if (features[2] <= 92006.5) {
                      if (features[7] <= 1.5) {
                        if (features[0] <= 92007.5) {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                        classes[3] = 0;
                      }
                    } else {
                      if (features[1] <= 102781.5) {
                        classes[0] = 7;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        if (features[2] <= 98228.5) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    }
                  } else {
                    if (features[7] <= 19.5) {
                      classes[0] = 70;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      if (features[7] <= 20.5) {
                        if (features[0] <= 91438.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 2;
                          classes[1] = 0;
                          classes[2] = 1;
                          classes[3] = 0;
                        }
                      } else {
                        classes[0] = 6;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    }
                  }
                } else {
                  if (features[0] <= 91959.5) {
                    classes[0] = 3;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 0;
                  } else {
                    if (features[2] <= 97625.0) {
                      if (features[2] <= 92849.5) {
                        if (features[0] <= 92050.5) {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 0;
                          classes[3] = 0;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 2;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    } else {
                      classes[0] = 1;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  }
                }
              }
            }
          } else {
            if (features[1] <= 112625.0) {
              if (features[1] <= 104210.0) {
                if (features[1] <= 104094.5) {
                  classes[0] = 0;
                  classes[1] = 3;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  classes[0] = 3;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 23;
                classes[2] = 0;
                classes[3] = 0;
              }
            } else {
              if (features[0] <= 120202.0) {
                classes[0] = 9;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 0;
              } else {
                if (features[7] <= 17.0) {
                  classes[0] = 0;
                  classes[1] = 12;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  if (features[8] <= 0.5) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 2;
                    classes[3] = 0;
                  } else {
                    classes[0] = 2;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  int index = 0;
  for (int i = 0; i < 4; i++) {
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