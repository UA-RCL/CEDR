#include <math.h>
#include <stdlib.h>
#define N_FEATURES 19
#define N_CLASSES 6

int lChilds[267] = {1,   2,   3,   4,   5,   -1,  7,   -1,  -1,  -1,  11,  12,  -1,  14,  -1,  -1,  17,  18,  19,  20,  -1,  -1,  -1,  -1,  25,  -1, -1,  28,  -1,  30,
                    -1,  32,  33,  34,  35,  36,  -1,  -1,  39,  40,  -1,  -1,  -1,  44,  45,  46,  -1,  -1,  49,  -1,  -1,  -1,  53,  54,  -1,  56, 57,  -1,  -1,  60,
                    -1,  -1,  63,  -1,  65,  66,  -1,  -1,  -1,  70,  71,  72,  73,  -1,  -1,  76,  -1,  78,  -1,  -1,  81,  82,  -1,  84,  -1,  -1, 87,  88,  -1,  -1,
                    -1,  92,  93,  94,  -1,  96,  -1,  -1,  99,  100, -1,  -1,  103, -1,  -1,  106, 107, -1,  109, -1,  -1,  112, -1,  114, -1,  -1, 117, 118, 119, 120,
                    -1,  122, -1,  -1,  125, -1,  127, -1,  129, -1,  131, -1,  -1,  134, -1,  136, 137, 138, -1,  140, 141, 142, -1,  -1,  -1,  -1, 147, 148, 149, 150,
                    -1,  -1,  -1,  -1,  155, 156, 157, -1,  -1,  -1,  161, 162, -1,  -1,  165, -1,  -1,  168, 169, -1,  171, -1,  173, -1,  175, -1, -1,  178, 179, -1,
                    181, -1,  183, -1,  -1,  186, 187, -1,  189, -1,  -1,  -1,  193, 194, 195, -1,  197, -1,  -1,  200, 201, 202, -1,  -1,  -1,  -1, 207, -1,  209, 210,
                    211, 212, -1,  214, -1,  -1,  -1,  218, -1,  220, -1,  -1,  223, 224, 225, 226, -1,  228, -1,  -1,  -1,  232, 233, 234, -1,  -1, -1,  238, -1,  240,
                    -1,  -1,  243, 244, 245, -1,  247, -1,  -1,  250, 251, -1,  -1,  -1,  255, 256, 257, -1,  -1,  260, -1,  -1,  263, -1,  265, -1, -1};
int rChilds[267] = {116, 27,  10,  9,   6,   -1,  8,   -1,  -1,  -1,  16,  13,  -1,  15,  -1,  -1,  24,  23,  22,  21,  -1,  -1,  -1,  -1,  26,  -1, -1,  29,  -1,  31,
                    -1,  69,  52,  43,  38,  37,  -1,  -1,  42,  41,  -1,  -1,  -1,  51,  48,  47,  -1,  -1,  50,  -1,  -1,  -1,  62,  55,  -1,  59, 58,  -1,  -1,  61,
                    -1,  -1,  64,  -1,  68,  67,  -1,  -1,  -1,  91,  80,  75,  74,  -1,  -1,  77,  -1,  79,  -1,  -1,  86,  83,  -1,  85,  -1,  -1, 90,  89,  -1,  -1,
                    -1,  105, 98,  95,  -1,  97,  -1,  -1,  102, 101, -1,  -1,  104, -1,  -1,  111, 108, -1,  110, -1,  -1,  113, -1,  115, -1,  -1, 192, 133, 124, 121,
                    -1,  123, -1,  -1,  126, -1,  128, -1,  130, -1,  132, -1,  -1,  135, -1,  167, 146, 139, -1,  145, 144, 143, -1,  -1,  -1,  -1, 154, 153, 152, 151,
                    -1,  -1,  -1,  -1,  160, 159, 158, -1,  -1,  -1,  164, 163, -1,  -1,  166, -1,  -1,  177, 170, -1,  172, -1,  174, -1,  176, -1, -1,  185, 180, -1,
                    182, -1,  184, -1,  -1,  191, 188, -1,  190, -1,  -1,  -1,  206, 199, 196, -1,  198, -1,  -1,  205, 204, 203, -1,  -1,  -1,  -1, 208, -1,  222, 217,
                    216, 213, -1,  215, -1,  -1,  -1,  219, -1,  221, -1,  -1,  242, 231, 230, 227, -1,  229, -1,  -1,  -1,  237, 236, 235, -1,  -1, -1,  239, -1,  241,
                    -1,  -1,  254, 249, 246, -1,  248, -1,  -1,  253, 252, -1,  -1,  -1,  262, 259, 258, -1,  -1,  261, -1,  -1,  264, -1,  266, -1, -1};
float thresholds[267] = {
    33.5,  5.0,  6.5,   75.0,   2.5,    -2.0,   2.5,   -2.0,  -2.0,   -2.0,   13.5, 9.5,  -2.0,   10.5, -2.0,   -2.0, 24.5, 0.5,    2.5,   8.5,    -2.0, -2.0, -2.0,  -2.0,  35.5,
    -2.0,  -2.0, 5.0,   -2.0,   0.5,    -2.0,   9.5,   3.5,   4.5,    1.5,    0.5,  -2.0, -2.0,   0.5,  1.5,    -2.0, -2.0, -2.0,   7.5,   2.0,    1.0,  -2.0, -2.0,  1.5,   -2.0,
    -2.0,  -2.0, 7.5,   2.0,    -2.0,   4.5,    4.5,   -2.0,  -2.0,   5.5,    -2.0, -2.0, 5.0,    -2.0, 8.5,    3.5,  -2.0, -2.0,   -2.0,  11.5,   5.5,  1.5,  0.5,   -2.0,  -2.0,
    2.5,   -2.0, 5.5,   -2.0,   -2.0,   8.5,    16.5,  -2.0,  6.5,    -2.0,   -2.0, 10.5, 9.0,    -2.0, -2.0,   -2.0, 24.5, 14.5,   12.5,  -2.0,   13.0, -2.0, -2.0,  20.5,  15.5,
    -2.0,  -2.0, 22.5,  -2.0,   -2.0,   23.5,   17.5,  -2.0,  17.0,   -2.0,   -2.0, 32.5, -2.0,   38.0, -2.0,   -2.0, 69.5, 5.0,    9.5,   1.5,    -2.0, 3.5,  -2.0,  -2.0,  13.0,
    -2.0,  23.5, -2.0,  1463.5, -2.0,   43.0,   -2.0,  -2.0,  5.0,    -2.0,   24.5, 0.5,  4.5,    -2.0, 9.0,    1.5,  1.5,  -2.0,   -2.0,  -2.0,   -2.0, 7.5,  4.5,   3.5,   4.0,
    -2.0,  -2.0, -2.0,  -2.0,   14.5,   33.0,   11.5,  -2.0,  -2.0,   -2.0,   0.5,  1.5,  -2.0,   -2.0, 1.5,    -2.0, -2.0, 44.5,   35.5,  -2.0,   34.0, -2.0, 36.0,  -2.0,  44.0,
    -2.0,  -2.0, 53.0,  43.5,   -2.0,   47.5,   -2.0,  54.5,  -2.0,   -2.0,   52.5, 51.5, -2.0,   7.5,  -2.0,   -2.0, -2.0, 5.0,    6.5,   2.5,    -2.0, 3.5,  -2.0,  -2.0,  14.5,
    9.5,   8.5,  -2.0,  -2.0,   -2.0,   -2.0,   5.0,   -2.0,  151.0,  65.0,   52.5, 42.5, -2.0,   10.5, -2.0,   -2.0, -2.0, 1980.5, -2.0,  70.0,   -2.0, -2.0, 942.5, 159.5, 135.0,
    86.5,  -2.0, 103.5, -2.0,   -2.0,   -2.0,   531.0, 242.5, 1771.0, -2.0,   -2.0, -2.0, 531.5,  -2.0, 562.5,  -2.0, -2.0, 1195.0, 988.0, 1076.0, -2.0, 0.5,  -2.0,  -2.0,  687.5,
    475.0, -2.0, -2.0,  -2.0,   1553.5, 1203.5, 760.5, -2.0,  -2.0,   1413.0, -2.0, -2.0, 1725.0, -2.0, 1807.5, -2.0, -2.0};
int indices[267] = {0,  7,  3,  5,  3,  -2, 4,  -2, -2, -2, 4,  4,  -2, 3,  -2, -2, 3,  2,  9,  0,  -2, -2, -2, -2, 3,  -2, -2, 8,  -2, 0,  -2, 1,  1,  14, 0,  1,  -2, -2, 14,
                    1,  -2, -2, -2, 14, 2,  15, -2, -2, 0,  -2, -2, -2, 0,  2,  -2, 1,  0,  -2, -2, 0,  -2, -2, 2,  -2, 0,  9,  -2, -2, -2, 2,  2,  0,  2,  -2, -2, 2,  -2, 0,
                    -2, -2, 0,  3,  -2, 2,  -2, -2, 0,  2,  -2, -2, -2, 1,  0,  0,  -2, 1,  -2, -2, 1,  0,  -2, -2, 0,  -2, -2, 2,  0,  -2, 2,  -2, -2, 0,  -2, 1,  -2, -2, 1,
                    7,  3,  3,  -2, 4,  -2, -2, 4,  -2, 3,  -2, 0,  -2, 4,  -2, -2, 8,  -2, 2,  1,  14, -2, 9,  2,  15, -2, -2, -2, -2, 2,  1,  2,  10, -2, -2, -2, -2, 1,  4,
                    1,  -2, -2, -2, 6,  5,  -2, -2, 15, -2, -2, 1,  1,  -2, 2,  -2, 0,  -2, 0,  -2, -2, 2,  0,  -2, 2,  -2, 1,  -2, -2, 0,  0,  -2, 9,  -2, -2, -2, 7,  3,  3,
                    -2, 4,  -2, -2, 4,  3,  3,  -2, -2, -2, -2, 8,  -2, 2,  0,  2,  5,  -2, 4,  -2, -2, -2, 0,  -2, 2,  -2, -2, 0,  1,  0,  0,  -2, 1,  -2, -2, -2, 0,  1,  2,
                    -2, -2, -2, 1,  -2, 2,  -2, -2, 1,  0,  2,  -2, 6,  -2, -2, 2,  1,  -2, -2, -2, 2,  0,  2,  -2, -2, 1,  -2, -2, 0,  -2, 1,  -2, -2};
int classes[267][6] = {{3941, 3289, 3092, 1474, 544, 1223},
                       {3585, 392, 275, 639, 238, 510},
                       {0, 0, 0, 639, 238, 0},
                       {0, 0, 0, 604, 5, 0},
                       {0, 0, 0, 604, 4, 0},
                       {0, 0, 0, 580, 0, 0},
                       {0, 0, 0, 24, 4, 0},
                       {0, 0, 0, 0, 4, 0},
                       {0, 0, 0, 24, 0, 0},
                       {0, 0, 0, 0, 1, 0},
                       {0, 0, 0, 35, 233, 0},
                       {0, 0, 0, 6, 226, 0},
                       {0, 0, 0, 0, 212, 0},
                       {0, 0, 0, 6, 14, 0},
                       {0, 0, 0, 6, 0, 0},
                       {0, 0, 0, 0, 14, 0},
                       {0, 0, 0, 29, 7, 0},
                       {0, 0, 0, 28, 1, 0},
                       {0, 0, 0, 4, 1, 0},
                       {0, 0, 0, 1, 1, 0},
                       {0, 0, 0, 0, 1, 0},
                       {0, 0, 0, 1, 0, 0},
                       {0, 0, 0, 3, 0, 0},
                       {0, 0, 0, 24, 0, 0},
                       {0, 0, 0, 1, 6, 0},
                       {0, 0, 0, 0, 6, 0},
                       {0, 0, 0, 1, 0, 0},
                       {3585, 392, 275, 0, 0, 510},
                       {0, 0, 0, 0, 0, 510},
                       {3585, 392, 275, 0, 0, 0},
                       {2280, 0, 0, 0, 0, 0},
                       {1305, 392, 275, 0, 0, 0},
                       {38, 348, 37, 0, 0, 0},
                       {8, 292, 24, 0, 0, 0},
                       {7, 232, 1, 0, 0, 0},
                       {6, 10, 0, 0, 0, 0},
                       {0, 10, 0, 0, 0, 0},
                       {6, 0, 0, 0, 0, 0},
                       {1, 222, 1, 0, 0, 0},
                       {1, 19, 1, 0, 0, 0},
                       {0, 16, 0, 0, 0, 0},
                       {1, 3, 1, 0, 0, 0},
                       {0, 203, 0, 0, 0, 0},
                       {1, 60, 23, 0, 0, 0},
                       {1, 21, 23, 0, 0, 0},
                       {0, 1, 23, 0, 0, 0},
                       {0, 1, 3, 0, 0, 0},
                       {0, 0, 20, 0, 0, 0},
                       {1, 20, 0, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 20, 0, 0, 0, 0},
                       {0, 39, 0, 0, 0, 0},
                       {30, 56, 13, 0, 0, 0},
                       {29, 5, 6, 0, 0, 0},
                       {0, 0, 6, 0, 0, 0},
                       {29, 5, 0, 0, 0, 0},
                       {4, 4, 0, 0, 0, 0},
                       {4, 0, 0, 0, 0, 0},
                       {0, 4, 0, 0, 0, 0},
                       {25, 1, 0, 0, 0, 0},
                       {22, 0, 0, 0, 0, 0},
                       {3, 1, 0, 0, 0, 0},
                       {1, 51, 7, 0, 0, 0},
                       {0, 0, 7, 0, 0, 0},
                       {1, 51, 0, 0, 0, 0},
                       {1, 3, 0, 0, 0, 0},
                       {0, 3, 0, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 48, 0, 0, 0, 0},
                       {1267, 44, 238, 0, 0, 0},
                       {29, 0, 224, 0, 0, 0},
                       {7, 0, 190, 0, 0, 0},
                       {5, 0, 13, 0, 0, 0},
                       {0, 0, 13, 0, 0, 0},
                       {5, 0, 0, 0, 0, 0},
                       {2, 0, 177, 0, 0, 0},
                       {0, 0, 154, 0, 0, 0},
                       {2, 0, 23, 0, 0, 0},
                       {2, 0, 1, 0, 0, 0},
                       {0, 0, 22, 0, 0, 0},
                       {22, 0, 34, 0, 0, 0},
                       {21, 0, 1, 0, 0, 0},
                       {20, 0, 0, 0, 0, 0},
                       {1, 0, 1, 0, 0, 0},
                       {0, 0, 1, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {1, 0, 33, 0, 0, 0},
                       {1, 0, 2, 0, 0, 0},
                       {0, 0, 2, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 0, 31, 0, 0, 0},
                       {1238, 44, 14, 0, 0, 0},
                       {77, 43, 0, 0, 0, 0},
                       {70, 4, 0, 0, 0, 0},
                       {62, 0, 0, 0, 0, 0},
                       {8, 4, 0, 0, 0, 0},
                       {0, 4, 0, 0, 0, 0},
                       {8, 0, 0, 0, 0, 0},
                       {7, 39, 0, 0, 0, 0},
                       {2, 33, 0, 0, 0, 0},
                       {1, 1, 0, 0, 0, 0},
                       {1, 32, 0, 0, 0, 0},
                       {5, 6, 0, 0, 0, 0},
                       {5, 0, 0, 0, 0, 0},
                       {0, 6, 0, 0, 0, 0},
                       {1161, 1, 14, 0, 0, 0},
                       {60, 0, 14, 0, 0, 0},
                       {55, 0, 0, 0, 0, 0},
                       {5, 0, 14, 0, 0, 0},
                       {0, 0, 9, 0, 0, 0},
                       {5, 0, 5, 0, 0, 0},
                       {1101, 1, 0, 0, 0, 0},
                       {1091, 0, 0, 0, 0, 0},
                       {10, 1, 0, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {10, 0, 0, 0, 0, 0},
                       {356, 2897, 2817, 835, 306, 713},
                       {15, 2784, 554, 476, 184, 407},
                       {0, 1, 0, 476, 184, 0},
                       {0, 0, 0, 463, 18, 0},
                       {0, 0, 0, 437, 0, 0},
                       {0, 0, 0, 26, 18, 0},
                       {0, 0, 0, 0, 18, 0},
                       {0, 0, 0, 26, 0, 0},
                       {0, 1, 0, 13, 166, 0},
                       {0, 0, 0, 0, 164, 0},
                       {0, 1, 0, 13, 2, 0},
                       {0, 0, 0, 12, 0, 0},
                       {0, 1, 0, 1, 2, 0},
                       {0, 0, 0, 0, 2, 0},
                       {0, 1, 0, 1, 0, 0},
                       {0, 0, 0, 1, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {15, 2783, 554, 0, 0, 407},
                       {0, 0, 0, 0, 0, 407},
                       {15, 2783, 554, 0, 0, 0},
                       {0, 952, 534, 0, 0, 0},
                       {0, 839, 97, 0, 0, 0},
                       {0, 692, 0, 0, 0, 0},
                       {0, 147, 97, 0, 0, 0},
                       {0, 20, 97, 0, 0, 0},
                       {0, 4, 97, 0, 0, 0},
                       {0, 2, 2, 0, 0, 0},
                       {0, 2, 95, 0, 0, 0},
                       {0, 16, 0, 0, 0, 0},
                       {0, 127, 0, 0, 0, 0},
                       {0, 113, 437, 0, 0, 0},
                       {0, 18, 364, 0, 0, 0},
                       {0, 18, 64, 0, 0, 0},
                       {0, 2, 64, 0, 0, 0},
                       {0, 1, 1, 0, 0, 0},
                       {0, 1, 63, 0, 0, 0},
                       {0, 16, 0, 0, 0, 0},
                       {0, 0, 300, 0, 0, 0},
                       {0, 95, 73, 0, 0, 0},
                       {0, 88, 4, 0, 0, 0},
                       {0, 88, 3, 0, 0, 0},
                       {0, 76, 0, 0, 0, 0},
                       {0, 12, 3, 0, 0, 0},
                       {0, 0, 1, 0, 0, 0},
                       {0, 7, 69, 0, 0, 0},
                       {0, 3, 2, 0, 0, 0},
                       {0, 3, 0, 0, 0, 0},
                       {0, 0, 2, 0, 0, 0},
                       {0, 4, 67, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {0, 3, 67, 0, 0, 0},
                       {15, 1831, 20, 0, 0, 0},
                       {2, 1679, 2, 0, 0, 0},
                       {0, 1633, 0, 0, 0, 0},
                       {2, 46, 2, 0, 0, 0},
                       {0, 0, 2, 0, 0, 0},
                       {2, 46, 0, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {1, 46, 0, 0, 0, 0},
                       {1, 3, 0, 0, 0, 0},
                       {0, 43, 0, 0, 0, 0},
                       {13, 152, 18, 0, 0, 0},
                       {1, 2, 18, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 2, 18, 0, 0, 0},
                       {0, 0, 14, 0, 0, 0},
                       {0, 2, 4, 0, 0, 0},
                       {0, 2, 0, 0, 0, 0},
                       {0, 0, 4, 0, 0, 0},
                       {12, 150, 0, 0, 0, 0},
                       {12, 1, 0, 0, 0, 0},
                       {10, 0, 0, 0, 0, 0},
                       {2, 1, 0, 0, 0, 0},
                       {2, 0, 0, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {0, 149, 0, 0, 0, 0},
                       {341, 113, 2263, 359, 122, 306},
                       {0, 0, 0, 359, 122, 0},
                       {0, 0, 0, 345, 3, 0},
                       {0, 0, 0, 331, 0, 0},
                       {0, 0, 0, 14, 3, 0},
                       {0, 0, 0, 0, 3, 0},
                       {0, 0, 0, 14, 0, 0},
                       {0, 0, 0, 14, 119, 0},
                       {0, 0, 0, 3, 119, 0},
                       {0, 0, 0, 3, 6, 0},
                       {0, 0, 0, 0, 6, 0},
                       {0, 0, 0, 3, 0, 0},
                       {0, 0, 0, 0, 113, 0},
                       {0, 0, 0, 11, 0, 0},
                       {341, 113, 2263, 0, 0, 306},
                       {0, 0, 0, 0, 0, 306},
                       {341, 113, 2263, 0, 0, 0},
                       {11, 1, 2206, 0, 0, 0},
                       {11, 0, 65, 0, 0, 0},
                       {1, 0, 65, 0, 0, 0},
                       {0, 0, 64, 0, 0, 0},
                       {1, 0, 1, 0, 0, 0},
                       {0, 0, 1, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {10, 0, 0, 0, 0, 0},
                       {0, 1, 2141, 0, 0, 0},
                       {0, 0, 2137, 0, 0, 0},
                       {0, 1, 4, 0, 0, 0},
                       {0, 0, 4, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {330, 112, 57, 0, 0, 0},
                       {301, 28, 4, 0, 0, 0},
                       {18, 19, 0, 0, 0, 0},
                       {18, 1, 0, 0, 0, 0},
                       {17, 0, 0, 0, 0, 0},
                       {1, 1, 0, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 18, 0, 0, 0, 0},
                       {283, 9, 4, 0, 0, 0},
                       {235, 1, 0, 0, 0, 0},
                       {18, 1, 0, 0, 0, 0},
                       {18, 0, 0, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {217, 0, 0, 0, 0, 0},
                       {48, 8, 4, 0, 0, 0},
                       {0, 8, 0, 0, 0, 0},
                       {48, 0, 4, 0, 0, 0},
                       {0, 0, 4, 0, 0, 0},
                       {48, 0, 0, 0, 0, 0},
                       {29, 84, 53, 0, 0, 0},
                       {1, 73, 6, 0, 0, 0},
                       {1, 2, 2, 0, 0, 0},
                       {0, 0, 2, 0, 0, 0},
                       {1, 2, 0, 0, 0, 0},
                       {1, 0, 0, 0, 0, 0},
                       {0, 2, 0, 0, 0, 0},
                       {0, 71, 4, 0, 0, 0},
                       {0, 11, 4, 0, 0, 0},
                       {0, 10, 0, 0, 0, 0},
                       {0, 1, 4, 0, 0, 0},
                       {0, 60, 0, 0, 0, 0},
                       {28, 11, 47, 0, 0, 0},
                       {6, 2, 47, 0, 0, 0},
                       {5, 0, 2, 0, 0, 0},
                       {0, 0, 2, 0, 0, 0},
                       {5, 0, 0, 0, 0, 0},
                       {1, 2, 45, 0, 0, 0},
                       {1, 2, 2, 0, 0, 0},
                       {0, 0, 43, 0, 0, 0},
                       {22, 9, 0, 0, 0, 0},
                       {20, 0, 0, 0, 0, 0},
                       {2, 9, 0, 0, 0, 0},
                       {0, 8, 0, 0, 0, 0},
                       {2, 1, 0, 0, 0, 0}};

int findMax(int nums[N_CLASSES]) {
  int index = 0;
  for (int i = 0; i < N_CLASSES; i++) {
    index = nums[i] > nums[index] ? i : index;
  }
  return index;
}

int predict(float features[N_FEATURES], int node) {
  if (thresholds[node] != -2) {
    if (features[indices[node]] <= thresholds[node]) {
      return predict(features, lChilds[node]);
    } else {
      return predict(features, rChilds[node]);
    }
  }
  return findMax(classes[node]);
}

// int main1(int argc, const char * argv[]) {
//
//     Features:
//    float features[argc-1];
//    int i;
//    for (i = 1; i < argc; i++) {
//        features[i-1] = atof(argv[i]);
//    }
//
//    Prediction:
//    printf("%d", predict(features, 0));
//    return 0;
//
//}