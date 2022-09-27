#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[9]) {

  int classes[3];

  if (features[0] <= 12717.0) {
    classes[0] = 48;
    classes[1] = 0;
    classes[2] = 0;
  } else {
    if (features[0] <= 17520.5) {
      if (features[1] <= 13506.0) {
        classes[0] = 0;
        classes[1] = 25;
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
              classes[1] = 2;
              classes[2] = 0;
            }
          } else {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 3;
          }
        } else {
          classes[0] = 0;
          classes[1] = 9;
          classes[2] = 0;
        }
      }
    } else {
      if (features[0] <= 30462.5) {
        classes[0] = 46;
        classes[1] = 0;
        classes[2] = 0;
      } else {
        if (features[2] <= 132539.0) {
          if (features[0] <= 90042.0) {
            if (features[0] <= 79418.5) {
              if (features[0] <= 48154.5) {
                if (features[1] <= 35783.5) {
                  if (features[1] <= 31729.5) {
                    classes[0] = 0;
                    classes[1] = 13;
                    classes[2] = 0;
                  } else {
                    classes[0] = 19;
                    classes[1] = 0;
                    classes[2] = 0;
                  }
                } else {
                  if (features[2] <= 43936.0) {
                    if (features[1] <= 42908.5) {
                      classes[0] = 0;
                      classes[1] = 12;
                      classes[2] = 0;
                    } else {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 13;
                    }
                  } else {
                    classes[0] = 0;
                    classes[1] = 28;
                    classes[2] = 0;
                  }
                }
              } else {
                if (features[0] <= 50856.5) {
                  if (features[7] <= 34.5) {
                    if (features[7] <= 4.5) {
                      if (features[0] <= 48919.5) {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                      } else {
                        classes[0] = 2;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 17;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  } else {
                    if (features[1] <= 53376.0) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 1;
                    } else {
                      classes[0] = 1;
                      classes[1] = 0;
                      classes[2] = 0;
                    }
                  }
                } else {
                  if (features[0] <= 52468.5) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 21;
                  } else {
                    if (features[0] <= 63600.5) {
                      if (features[1] <= 58180.0) {
                        if (features[0] <= 58797.0) {
                          classes[0] = 4;
                          classes[1] = 0;
                          classes[2] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 5;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 28;
                        classes[2] = 0;
                      }
                    } else {
                      if (features[0] <= 69762.0) {
                        classes[0] = 22;
                        classes[1] = 0;
                        classes[2] = 0;
                      } else {
                        if (features[2] <= 70286.0) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 13;
                        } else {
                          classes[0] = 21;
                          classes[1] = 21;
                          classes[2] = 18;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if (features[0] <= 81901.0) {
                classes[0] = 22;
                classes[1] = 0;
                classes[2] = 0;
              } else {
                if (features[2] <= 86531.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 2;
                } else {
                  classes[0] = 9;
                  classes[1] = 0;
                  classes[2] = 0;
                }
              }
            }
          } else {
            if (features[1] <= 130178.5) {
              if (features[1] <= 90966.5) {
                classes[0] = 0;
                classes[1] = 26;
                classes[2] = 0;
              } else {
                if (features[2] <= 97809.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 14;
                } else {
                  if (features[0] <= 114372.0) {
                    if (features[1] <= 108835.5) {
                      if (features[0] <= 101805.0) {
                        if (features[0] <= 96426.0) {
                          classes[0] = 7;
                          classes[1] = 18;
                          classes[2] = 0;
                        } else {
                          classes[0] = 4;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      } else {
                        if (features[2] <= 102621.0) {
                          classes[0] = 0;
                          classes[1] = 9;
                          classes[2] = 2;
                        } else {
                          classes[0] = 0;
                          classes[1] = 15;
                          classes[2] = 0;
                        }
                      }
                    } else {
                      if (features[0] <= 107011.5) {
                        classes[0] = 8;
                        classes[1] = 0;
                        classes[2] = 0;
                      } else {
                        if (features[0] <= 109082.0) {
                          classes[0] = 4;
                          classes[1] = 0;
                          classes[2] = 20;
                        } else {
                          classes[0] = 3;
                          classes[1] = 0;
                          classes[2] = 0;
                        }
                      }
                    }
                  } else {
                    if (features[0] <= 126720.5) {
                      if (features[1] <= 120911.0) {
                        if (features[0] <= 119175.5) {
                          classes[0] = 0;
                          classes[1] = 14;
                          classes[2] = 0;
                        } else {
                          classes[0] = 5;
                          classes[1] = 12;
                          classes[2] = 2;
                        }
                      } else {
                        classes[0] = 5;
                        classes[1] = 0;
                        classes[2] = 0;
                      }
                    } else {
                      classes[0] = 0;
                      classes[1] = 20;
                      classes[2] = 0;
                    }
                  }
                }
              }
            } else {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 10;
            }
          }
        } else {
          if (features[1] <= 134966.5) {
            classes[0] = 36;
            classes[1] = 0;
            classes[2] = 0;
          } else {
            classes[0] = 0;
            classes[1] = 7;
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