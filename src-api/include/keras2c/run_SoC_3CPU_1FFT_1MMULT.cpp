#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[10]) {

  int classes[4];

  if (features[0] <= 43351.0) {
    if (features[0] <= 12717.0) {
      classes[0] = 50;
      classes[1] = 0;
      classes[2] = 0;
      classes[3] = 0;
    } else {
      if (features[0] <= 17520.5) {
        if (features[1] <= 13506.0) {
          classes[0] = 0;
          classes[1] = 26;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[3] <= 7405.5) {
            if (features[9] <= 5.0) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 0;
              classes[3] = 1;
            } else {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 1;
              classes[3] = 0;
            }
          } else {
            if (features[1] <= 17267.5) {
              classes[0] = 0;
              classes[1] = 6;
              classes[2] = 0;
              classes[3] = 0;
            } else {
              if (features[8] <= 6.5) {
                classes[0] = 1;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 0;
              } else {
                classes[0] = 0;
                classes[1] = 3;
                classes[2] = 0;
                classes[3] = 0;
              }
            }
          }
        }
      } else {
        if (features[0] <= 30461.5) {
          classes[0] = 42;
          classes[1] = 0;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[1] <= 31729.5) {
            classes[0] = 0;
            classes[1] = 14;
            classes[2] = 0;
            classes[3] = 0;
          } else {
            if (features[8] <= 0.5) {
              if (features[3] <= 32012.0) {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 2;
              } else {
                classes[0] = 1;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 0;
              }
            } else {
              classes[0] = 19;
              classes[1] = 0;
              classes[2] = 0;
              classes[3] = 0;
            }
          }
        }
      }
    }
  } else {
    if (features[0] <= 48154.5) {
      if (features[3] <= 38357.0) {
        if (features[2] <= 29865.0) {
          classes[0] = 0;
          classes[1] = 15;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[2] <= 43936.0) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 14;
            classes[3] = 0;
          } else {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 0;
            classes[3] = 1;
          }
        }
      } else {
        classes[0] = 0;
        classes[1] = 28;
        classes[2] = 0;
        classes[3] = 0;
      }
    } else {
      if (features[0] <= 50856.5) {
        if (features[8] <= 4.5) {
          if (features[1] <= 53376.5) {
            if (features[0] <= 48582.0) {
              if (features[8] <= 0.5) {
                classes[0] = 0;
                classes[1] = 1;
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
        } else {
          classes[0] = 24;
          classes[1] = 0;
          classes[2] = 0;
          classes[3] = 0;
        }
      } else {
        if (features[2] <= 60250.0) {
          if (features[2] <= 51939.0) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 22;
            classes[3] = 0;
          } else {
            if (features[0] <= 58797.0) {
              classes[0] = 3;
              classes[1] = 0;
              classes[2] = 0;
              classes[3] = 0;
            } else {
              if (features[8] <= 3.0) {
                if (features[3] <= 53932.5) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 1;
                } else {
                  if (features[8] <= 1.0) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                    classes[3] = 0;
                  } else {
                    if (features[3] <= 55411.5) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 1;
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                      classes[3] = 0;
                    }
                  }
                }
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 4;
                classes[3] = 0;
              }
            }
          }
        } else {
          if (features[1] <= 68297.0) {
            if (features[8] <= 17.5) {
              if (features[8] <= 0.5) {
                if (features[1] <= 62196.5) {
                  classes[0] = 0;
                  classes[1] = 2;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 1;
                }
              } else {
                classes[0] = 0;
                classes[1] = 20;
                classes[2] = 0;
                classes[3] = 0;
              }
            } else {
              if (features[8] <= 21.5) {
                if (features[9] <= 0.5) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 2;
                } else {
                  classes[0] = 0;
                  classes[1] = 1;
                  classes[2] = 0;
                  classes[3] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 5;
                classes[2] = 0;
                classes[3] = 0;
              }
            }
          } else {
            if (features[2] <= 65053.5) {
              classes[0] = 18;
              classes[1] = 0;
              classes[2] = 0;
              classes[3] = 0;
            } else {
              if (features[0] <= 107169.5) {
                if (features[0] <= 90201.0) {
                  if (features[0] <= 79559.5) {
                    if (features[1] <= 80882.0) {
                      if (features[2] <= 75653.5) {
                        if (features[1] <= 73094.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 8;
                          classes[3] = 1;
                        } else {
                          classes[0] = 5;
                          classes[1] = 1;
                          classes[2] = 2;
                          classes[3] = 1;
                        }
                      } else {
                        if (features[3] <= 75666.0) {
                          classes[0] = 0;
                          classes[1] = 14;
                          classes[2] = 0;
                          classes[3] = 2;
                        } else {
                          classes[0] = 7;
                          classes[1] = 7;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    } else {
                      if (features[0] <= 78888.0) {
                        if (features[3] <= 77423.0) {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 16;
                          classes[3] = 1;
                        } else {
                          classes[0] = 2;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 13;
                        classes[3] = 0;
                      }
                    }
                  } else {
                    if (features[0] <= 82058.5) {
                      classes[0] = 17;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      if (features[0] <= 83002.0) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 2;
                        classes[3] = 0;
                      } else {
                        classes[0] = 7;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    }
                  }
                } else {
                  if (features[8] <= 15.5) {
                    if (features[1] <= 108994.5) {
                      if (features[1] <= 90989.0) {
                        classes[0] = 0;
                        classes[1] = 17;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        if (features[2] <= 97964.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 6;
                          classes[3] = 0;
                        } else {
                          classes[0] = 3;
                          classes[1] = 30;
                          classes[2] = 1;
                          classes[3] = 2;
                        }
                      }
                    } else {
                      classes[0] = 2;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  } else {
                    if (features[1] <= 95856.0) {
                      if (features[1] <= 90972.5) {
                        if (features[8] <= 18.5) {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 0;
                          classes[3] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 2;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        if (features[8] <= 21.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 2;
                          classes[3] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 4;
                          classes[3] = 0;
                        }
                      }
                    } else {
                      if (features[0] <= 101683.0) {
                        if (features[8] <= 21.5) {
                          classes[0] = 3;
                          classes[1] = 3;
                          classes[2] = 0;
                          classes[3] = 2;
                        } else {
                          classes[0] = 4;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      } else {
                        if (features[8] <= 22.0) {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 3;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    }
                  }
                }
              } else {
                if (features[2] <= 108637.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 20;
                  classes[3] = 0;
                } else {
                  if (features[3] <= 108200.5) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 1;
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