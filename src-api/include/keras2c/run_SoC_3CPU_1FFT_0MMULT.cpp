#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int predict(float features[9]) {

  int classes[4];

  if (features[0] <= 43351.0) {
    if (features[0] <= 12717.0) {
      classes[0] = 55;
      classes[1] = 0;
      classes[2] = 0;
      classes[3] = 0;
    } else {
      if (features[0] <= 17520.5) {
        if (features[2] <= 15774.5) {
          classes[0] = 0;
          classes[1] = 28;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[3] <= 15919.0) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 1;
            classes[3] = 0;
          } else {
            if (features[7] <= 6.5) {
              if (features[1] <= 17267.5) {
                classes[0] = 0;
                classes[1] = 2;
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
              classes[1] = 4;
              classes[2] = 0;
              classes[3] = 0;
            }
          }
        }
      } else {
        if (features[0] <= 30461.5) {
          classes[0] = 48;
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
            if (features[7] <= 0.5) {
              if (features[0] <= 33443.5) {
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
              classes[0] = 20;
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
        if (features[1] <= 42915.0) {
          classes[0] = 0;
          classes[1] = 13;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[2] <= 43936.0) {
            classes[0] = 0;
            classes[1] = 0;
            classes[2] = 10;
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
        classes[1] = 25;
        classes[2] = 0;
        classes[3] = 0;
      }
    } else {
      if (features[0] <= 90201.0) {
        if (features[3] <= 78888.0) {
          if (features[0] <= 50856.5) {
            if (features[7] <= 3.5) {
              if (features[0] <= 48582.0) {
                if (features[1] <= 48573.5) {
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
                classes[1] = 1;
                classes[2] = 0;
                classes[3] = 0;
              }
            } else {
              if (features[7] <= 5.5) {
                if (features[1] <= 53376.5) {
                  classes[0] = 0;
                  classes[1] = 1;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  classes[0] = 2;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 0;
                }
              } else {
                classes[0] = 18;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 0;
              }
            }
          } else {
            if (features[2] <= 51939.0) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 25;
              classes[3] = 0;
            } else {
              if (features[3] <= 76106.0) {
                if (features[0] <= 63600.5) {
                  if (features[1] <= 58180.0) {
                    if (features[0] <= 58797.0) {
                      if (features[7] <= 0.5) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 1;
                      } else {
                        classes[0] = 3;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    } else {
                      if (features[3] <= 55411.5) {
                        if (features[3] <= 53932.5) {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 1;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 4;
                        classes[3] = 0;
                      }
                    }
                  } else {
                    if (features[7] <= 0.5) {
                      if (features[1] <= 62196.5) {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 1;
                      }
                    } else {
                      if (features[7] <= 17.5) {
                        classes[0] = 0;
                        classes[1] = 18;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        if (features[7] <= 21.5) {
                          classes[0] = 0;
                          classes[1] = 1;
                          classes[2] = 0;
                          classes[3] = 2;
                        } else {
                          classes[0] = 0;
                          classes[1] = 6;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    }
                  }
                } else {
                  if (features[0] <= 69762.0) {
                    classes[0] = 18;
                    classes[1] = 0;
                    classes[2] = 0;
                    classes[3] = 0;
                  } else {
                    if (features[2] <= 70263.0) {
                      if (features[7] <= 19.5) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 10;
                        classes[3] = 0;
                      } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 1;
                      }
                    } else {
                      if (features[0] <= 75817.5) {
                        if (features[2] <= 75653.5) {
                          classes[0] = 2;
                          classes[1] = 1;
                          classes[2] = 0;
                          classes[3] = 1;
                        } else {
                          classes[0] = 0;
                          classes[1] = 10;
                          classes[2] = 0;
                          classes[3] = 1;
                        }
                      } else {
                        if (features[0] <= 76001.5) {
                          classes[0] = 7;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 0;
                        } else {
                          classes[0] = 4;
                          classes[1] = 9;
                          classes[2] = 0;
                          classes[3] = 0;
                        }
                      }
                    }
                  }
                }
              } else {
                if (features[0] <= 78888.0) {
                  if (features[0] <= 76638.5) {
                    if (features[2] <= 76331.0) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 3;
                      classes[3] = 0;
                    } else {
                      if (features[7] <= 21.0) {
                        if (features[7] <= 17.5) {
                          classes[0] = 1;
                          classes[1] = 0;
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
                        classes[1] = 0;
                        classes[2] = 2;
                        classes[3] = 0;
                      }
                    }
                  } else {
                    if (features[2] <= 76932.0) {
                      classes[0] = 3;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      if (features[3] <= 77423.0) {
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
                } else {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 11;
                  classes[3] = 0;
                }
              }
            }
          }
        } else {
          if (features[0] <= 82059.5) {
            classes[0] = 21;
            classes[1] = 0;
            classes[2] = 0;
            classes[3] = 0;
          } else {
            if (features[3] <= 81272.0) {
              classes[0] = 0;
              classes[1] = 0;
              classes[2] = 1;
              classes[3] = 0;
            } else {
              classes[0] = 11;
              classes[1] = 0;
              classes[2] = 0;
              classes[3] = 0;
            }
          }
        }
      } else {
        if (features[1] <= 90989.0) {
          classes[0] = 0;
          classes[1] = 26;
          classes[2] = 0;
          classes[3] = 0;
        } else {
          if (features[2] <= 97964.0) {
            if (features[7] <= 17.5) {
              if (features[7] <= 2.5) {
                if (features[2] <= 93140.0) {
                  classes[0] = 0;
                  classes[1] = 0;
                  classes[2] = 3;
                  classes[3] = 0;
                } else {
                  if (features[3] <= 93117.0) {
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
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 7;
                classes[3] = 0;
              }
            } else {
              if (features[7] <= 22.0) {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 0;
                classes[3] = 2;
              } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 3;
                classes[3] = 0;
              }
            }
          } else {
            if (features[1] <= 108994.5) {
              if (features[0] <= 96460.0) {
                if (features[1] <= 96072.0) {
                  classes[0] = 0;
                  classes[1] = 6;
                  classes[2] = 0;
                  classes[3] = 0;
                } else {
                  if (features[7] <= 13.0) {
                    if (features[7] <= 3.0) {
                      if (features[3] <= 96205.0) {
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
                    } else {
                      classes[0] = 0;
                      classes[1] = 3;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  } else {
                    if (features[0] <= 96194.5) {
                      if (features[7] <= 34.0) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 1;
                      } else {
                        classes[0] = 1;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    } else {
                      classes[0] = 5;
                      classes[1] = 0;
                      classes[2] = 0;
                      classes[3] = 0;
                    }
                  }
                }
              } else {
                if (features[7] <= 0.5) {
                  if (features[3] <= 96404.5) {
                    classes[0] = 0;
                    classes[1] = 1;
                    classes[2] = 0;
                    classes[3] = 0;
                  } else {
                    if (features[3] <= 98469.0) {
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
                } else {
                  if (features[1] <= 101913.5) {
                    if (features[0] <= 96602.5) {
                      classes[0] = 0;
                      classes[1] = 7;
                      classes[2] = 0;
                      classes[3] = 0;
                    } else {
                      if (features[1] <= 96911.5) {
                        classes[0] = 2;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 0;
                      } else {
                        classes[0] = 0;
                        classes[1] = 6;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    }
                  } else {
                    if (features[2] <= 102777.0) {
                      classes[0] = 0;
                      classes[1] = 0;
                      classes[2] = 2;
                      classes[3] = 0;
                    } else {
                      if (features[1] <= 103170.5) {
                        if (features[1] <= 103019.5) {
                          classes[0] = 0;
                          classes[1] = 4;
                          classes[2] = 1;
                          classes[3] = 0;
                        } else {
                          classes[0] = 0;
                          classes[1] = 0;
                          classes[2] = 0;
                          classes[3] = 1;
                        }
                      } else {
                        classes[0] = 0;
                        classes[1] = 7;
                        classes[2] = 0;
                        classes[3] = 0;
                      }
                    }
                  }
                }
              }
            } else {
              if (features[0] <= 114531.0) {
                if (features[2] <= 108637.0) {
                  if (features[0] <= 107169.5) {
                    if (features[7] <= 19.5) {
                      classes[0] = 7;
                      classes[1] = 0;
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
                    classes[1] = 0;
                    classes[2] = 12;
                    classes[3] = 0;
                  }
                } else {
                  classes[0] = 7;
                  classes[1] = 0;
                  classes[2] = 0;
                  classes[3] = 0;
                }
              } else {
                classes[0] = 0;
                classes[1] = 7;
                classes[2] = 0;
                classes[3] = 0;
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