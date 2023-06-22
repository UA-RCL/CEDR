// C++ program to find adjoint and inverse of a matrix
//#include<bits/stdc++.h>
#include "inverse.h"

// Function to calculate and store inverse, returns false if
// matrix is singular
void inverse(float A[N*N], float inverse[N*N])
{
//#pragma HLS inline region recursive
    // Find determinant of A[][]
    float det = determinant(A, N);
    cout<<"Value of determinant is "<<det;
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return;
    }

    // Find adjoint
    float adj[N][N],result=0.0;
    adjoint(A, adj);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
    {

// #pragma HLS loop_tripcount min=4 max=10
        for (int j=0; j<N; j++)
        {

// #pragma HLS loop_tripcount min=4 max=10
        	result=divide(adj[i][j],det);
//        	cout<<"Result from division of "<<adj[i][j]<<"with"<<det<<"is"<<result;
//        	g=adj[i][j]/det;
//#pragma HLS RESOURCE variable=g core=FDiv
        	inverse[i*N+j] = result;
        }
    }
//    return true;
    cout<<"Returning from the inverse function";
}

