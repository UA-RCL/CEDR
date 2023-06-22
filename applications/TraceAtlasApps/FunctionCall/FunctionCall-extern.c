#define WIDTH 1024
#include <stdlib.h>

void Kernel0(int* output, int* input)
{
    output[0] = abs(input[0]);
}

int Kernel1(int input)
{
    return input * 3 - 2;
}

void Kernel2(int* input, int* output)
{
    for(int i = 0; i < WIDTH; i++)
    {
        output[i] = input[i] * -1;
    }
}