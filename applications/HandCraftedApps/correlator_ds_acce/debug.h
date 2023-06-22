//
// Created by sahilhassan on 1/21/21.
//
#include <stdio.h>

//void PRINT_ARRAY(float *array, int size, char *array_name, char* func_name, int iter) {
void PRINT_ARRAY(double *array, int size, char *array_name, char* func_name, int iter) {
  printf("[%s %d] ========================================= DEBUG PRINT ===============================================\n", func_name, iter);
  printf("[%s %d] Value in %s is\n", func_name, iter, array_name);
  for (int i=0; i < size; i+=2){
    printf("(%lf, %lf i), ", array[i], array[i+1]);
  }
  printf("\n[%s %d] =================================================================================================\n", func_name, iter);
}