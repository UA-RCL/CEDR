#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[8]) {

  int classes[3];

  if (features[0] <= 12717.0) {
    classes[0] = 53;
    classes[1] = 0;
    classes[2] = 0;
  } else {
    if (features[2] <= 132524.5) {
      if (features[0] <= 90042.0) {
        if (features[0] <= 17521.0) {
          if (features[1] <= 13506.0) {
            classes[0] = 0;
            classes[1] = 28;
            classes[2] = 0;
          } else {
            if (features[2] <= 15793.5) {
              if (features[1] <= 15437.5) {
                if (features[2] <= 7887.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 1;
                } else {
                  classes[0] = 0;
                  classes[1] = 1;
                  classes[2] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 3;
              }
            } else {
              classes[0] = 0;
              classes[1] = 7;
              classes[2] = 0;
            }
          }
        } else {
          if (features[0] <= 30462.5) {
            classes[0] = 44;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            if (features[1] <= 68284.0) {
              if (features[1] <= 58180.0) {
                if (features[0] <= 50855.5) {
                  if (features[0] <= 48155.0) {
                    if (features[0] <= 43351.0) {
                      if (features[1] <= 31729.5) {
                        classes[0] = 0;
                        classes[1] = 11;
                        classes[2] = 0;
                      } else {
                        classes[0] = 18;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      if (features[2] <= 43936.0) {
                        if (features[2] <= 29865.5) {
                          classes[0] = 0;
                          classes[1] = 11;
                          classes[2] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 18;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 22;
                        classes[2] = 0;
                      }
                    }
                  } else {
                    if (features[6] <= 37.5) {
                      if (features[6] <= 5.5) {
                        if (features[1] <= 53376.5) {
                          classes[0] = 0;
                          classes[1] = 2;
                          classes[2] = 0;
                        } else {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 16;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                    }
                  }
                } else {
                  if (features[0] <= 52468.5) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 19;
                  } else {
                    if (features[0] <= 58797.0) {
                      classes[0] = 6;
                      classes[1] = 0;
                      classes[2] = 0;
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 5;
                    }
                  }
                }
              } else {
                classes[0] = 0;
                classes[1] = 22;
                classes[2] = 0;
              }
            } else {
              if (features[0] <= 69762.0) {
                classes[0] = 27;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                if (features[0] <= 79401.0) {
                  if (features[1] <= 80845.5) {
                    if (features[2] <= 70286.0) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 12;
                    } else {
                      if (features[1] <= 75810.0) {
                        if (features[2] <= 75678.0) {
                          classes[0] = 2;
                          classes[1] = 1;
                          classes[2] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 12;
                          classes[2] = 0;
                        }
                      } else {
                        if (features[0] <= 75970.0) {
                          classes[0] = 10;
                          classes[1] = 0;
                          classes[2] = 0;
                        } else {
                          classes[0] = 2;
                          classes[1] = 5;
                          classes[2] = 0;
                        }
                      }
                    }
                  } else {
                    if (features[0] <= 78729.5) {
                      if (features[2] <= 76500.0) {
                        if (features[6] <= 19.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 7;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 1;
                        }
                      } else {
                        if (features[0] <= 77359.0) {
                          classes[0] = 5;
                          classes[1] = 0;
                          classes[2] = 0;
                        } else {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 5;
                        }
                      }
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 11;
                    }
                  }
                } else {
                  if (features[0] <= 81901.0) {
                    classes[0] = 21;
                    classes[1] = 0;
                    classes[2] = 0;
                  } else {
                    if (features[2] <= 86531.5) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 4;
                    } else {
                      classes[0] = 9;
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
        if (features[2] <= 122392.0) {
          if (features[2] <= 91922.0) {
            classes[0] = 0;
            classes[1] = 25;
            classes[2] = 0;
          } else {
            if (features[2] <= 97809.0) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 12;
            } else {
              if (features[1] <= 108835.0) {
                if (features[0] <= 101805.0) {
                  if (features[1] <= 96216.5) {
                    if (features[6] <= 22.5) {
                      if (features[0] <= 96237.0) {
                        if (features[1] <= 96062.0) {
                          classes[0] = 0;
                          classes[1] = 8;
                          classes[2] = 0;
                        } else {
                          classes[0] = 4;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 12;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 2;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  } else {
                    classes[0] = 6;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                } else {
                  if (features[1] <= 102859.5) {
                    if (features[1] <= 102671.0) {
                      classes[0] = 0;
                      classes[1] = 9;
                      classes[2] = 0;
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                    }
                  } else {
                    classes[0] = 0;
                    classes[1] = 17;
                    classes[2] = 0;
                  }
                }
              } else {
                if (features[0] <= 114372.0) {
                  if (features[0] <= 107012.0) {
                    classes[0] = 7;
                    classes[1] = 0;
                    classes[2] = 0;
                  } else {
                    if (features[6] <= 23.0) {
                      if (features[2] <= 108478.0) {
                        if (features[6] <= 21.5) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 16;
                        } else {
                          classes[0] = 1;
                          classes[1] = 0;
                          classes[2] = 1;
                        }
                      } else {
                        classes[0] = 3;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 3;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  }
                } else {
                  if (features[1] <= 122340.5) {
                    if (features[1] <= 120911.0) {
                      if (features[0] <= 119175.5) {
                        classes[0] = 0;
                        classes[1] = 15;
                        classes[2] = 0;
                      } else {
                        if (features[1] <= 120430.5) {
                          classes[0] = 8;
                          classes[1] = 1;
                          classes[2] = 2;
                        } else {
                          classes[0] = 0;
                          classes[1] = 11;
                          classes[2] = 0;
                        }
                      }
                    } else {
                      classes[0] = 6;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  } else {
                    classes[0] = 0;
                    classes[1] = 17;
                    classes[2] = 0;
                  }
                }
              }
            }
          }
        } else {
          classes[0] = 0;
          classes[1] = 0;
          classes[2] = 8;
        }
      }
    } else {
      classes[0] = 36;
      classes[1] = 0;
      classes[2] = 0;
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