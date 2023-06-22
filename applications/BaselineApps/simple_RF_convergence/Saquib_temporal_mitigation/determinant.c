#include "inverse.h"

float determinant(float A[N*N], int n)
{
//#pragma HLS inline region recursive
    float D = 0; // Initialize result

    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0];

    float temp[N*N]; // To store cofactors

    int sign = 1;  // To store sign multiplier

     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
//#pragma HLS loop_tripcount min=16 max=72
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0 + f] * determinant(temp, n - 1);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}
