#include <cstdio>
#include <cstdlib>

extern "C" void Allocation(int** x, int** y, int** z, int* num_elem) {
    *num_elem = 32000;
    (*x) = (int*) calloc(*num_elem, sizeof(int));
    (*y) = (int*) calloc(*num_elem, sizeof(int));
    (*z) = (int*) calloc(*num_elem, sizeof(int));
}

extern "C" void Initialization(int** x, int** y, int* num_elem) {
    for (int i = 0; i < *num_elem; i++) {
        (*x)[i] = i;
        (*y)[i] = i;
    }
    printf("After initialization this is X:\n[");
    for (int i = 0; i < *num_elem; i++) {
        printf("%d ", (*x)[i]);
    }
    printf("]\n");
    printf("And this is Y:\n[");
    for (int i = 0; i < *num_elem; i++) {
        printf("%d ", (*y)[i]);
    }
    printf("]\n");
}

extern "C" void Vector_Add_CPU(int** x, int** y, int** z, int* num_elem) {
    printf("---------------------------------------\n");
    printf("------- Vector Addition on CPU --------\n");
    printf("---------------------------------------\n");
    for (int i = 0; i < *num_elem; i++) {
        (*z)[i] = (*x)[i] + (*y)[i];
    }
}

extern "C" void Print_Result(int** z, int* num_elem) {
    printf("Printing the resulting sum vector\n[");
    for (int i = 0; i < *num_elem; i++) {
        printf("%d ", (*z)[i]);
    }
    printf("]\n");
}