#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#define WIDTH 1024

void Kernel0(int* output, int* input);
int Kernel1(int input);
void Kernel2(int* input, int* output);

int main()
{
    int *buf0, *buf1, *buf2, *buf3;
    int i;

    for (i = 0; i < 5; i++) {}

    buf0 = (int*) malloc(sizeof(int) * WIDTH);
    buf1 = (int*) malloc(sizeof(int) * WIDTH);
    buf2 = (int*) malloc(sizeof(int) * WIDTH);
    buf3 = (int*) malloc(sizeof(int) * WIDTH);

    srand(time(NULL));
    //initialize the data
    for(i = 0; i < WIDTH; i++)
    {
        buf0[i] = rand();
    }
    
    for(i = 0; i < WIDTH; i++)
    {
        Kernel0(&(buf1[i]), &(buf0[i]));
    }

    for(i = 0; i < WIDTH; i++)
    {
        buf2[i] = Kernel1(buf0[i]);
    }

    Kernel2(buf2, buf3);

    printf("Success\n");
    return 0;
}

__attribute__((always_inline)) void Kernel0(int* output, int* input)
{
    output[0] = abs(input[0]);
}

__attribute__((always_inline)) int Kernel1(int input)
{
    return input * 3 - 2;
}

__attribute__((always_inline)) void Kernel2(int* input, int* output)
{
    for(int i = 0; i < WIDTH; i++)
    {
        output[i] = input[i] * -1;
    }
}