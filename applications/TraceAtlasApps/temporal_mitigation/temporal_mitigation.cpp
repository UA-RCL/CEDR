#include "inverse.h"
#include <ctime>

#define PROGPATH "./input/"
#define ZIN PROGPATH "z.txt"
#define ZIMAGIN PROGPATH "zimag.txt"
#define SIN PROGPATH "s.txt"
#define SIMAGIN PROGPATH "simag.txt"

#include "DashExtras.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" int main(void) {
    // Initializing the main variables that will be required in the overall computation of the Z response

    int i, j, k, m;

    float *Z, *Zi;
    float *S, *Si;

    FILE *Zreal, *Zimag, *Sreal, *Simag;

    float *Shermitian, *Shermitianimag;

    float res1 = 0, res2 = 0, res3 = 0, res4 = 0;
    float term1, term2, term3, term4;

    float *result1, *result1imag;

    float *result2, *result2imag;

    float *inv1, *inv2, *intmedt1, *intmedt2, *intmedb1, *intmedb2, *intmedb3, *intmedb4;
    float *buffer1, *buffer2, *buffer3, *buffer4;
    float *bufferinv1, *bufferinv2, *bufferinv3, *bufferinv4, *bufferinv5;

    float *result3, *result3imag;

    float *result4, *result4imag;

    float *zres, *zresimag;

    clock_t begin, end;

    for (i = 0; i < 1; i++) {}

    begin = clock();

    // Initializing the Z signal which will have 4*64 dimension
    Z = (float *)malloc(N * M * sizeof(float));
    Zi = (float *)malloc(N * M * sizeof(float));

    // Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted
    // by S
    S = (float *)malloc(N * M * sizeof(float));
    Si = (float *)malloc(N * M * sizeof(float));

    // now defining the argument files which will contain the corresponding values of Z and S
    Zreal = fopen(ZIN, "r");

    if (Zreal == nullptr) {
      printf("Zreal was null! Printing errno message: %s\n", strerror(errno));
      printf("I'm going to call exit(1) now\n");
      exit(1);
    }

    Zimag = fopen(ZIMAGIN, "r");
    
    if (Zimag == nullptr) {
      printf("Zimag was null! Printing errno message: %s\n", strerror(errno));
      printf("I'm going to call exit(1) now\n");
      exit(1);
    }

    Sreal = fopen(SIN, "r");

    if (Sreal == nullptr) {
      printf("Sreal was null! Printing errno message: %s\n", strerror(errno));
      printf("I'm going to call exit(1) now\n");
      exit(1);
    }

    Simag = fopen(SIMAGIN, "r");

    if (Simag == nullptr) {
      printf("Simag was null! Printing errno message: %s\n", strerror(errno));
      printf("I'm going to call exit(1) now\n");
      exit(1);
    }

    //printf("If the program has made it here, all files were opened successfully!\n");

    // now copying the contents of the files into the arrays that have been assigned for the signal and the jammer
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            fscanf(Zreal, "%f", &Z[i * M + j]); Z[i * M + j] /= 10.0f;
            fscanf(Zimag, "%f", &Zi[i * M + j]); Zi[i * M + j] /= 10.0f;
            fscanf(Sreal, "%f", &S[i * M + j]); S[i * M + j] /= 10.0f;
            fscanf(Simag, "%f", &Si[i * M + j]); Si[i * M + j] /= 10.0f;
        }
    }

    //cout << "Done reading the files from the input provided \n";
    //// Computing the hermitian of S
    Shermitian = (float *)malloc(M * N * sizeof(float));
    Shermitianimag = (float *)malloc(M * N * sizeof(float));

    //cout << "Calling a function to compute the hermitian \n";

    KERN_ENTER(make_label("transpose[Ar-%d][Ac-%d][complex][float32]", N, M));
    hermitian(S, Si, Shermitian, Shermitianimag);
    KERN_EXIT(make_label("transpose[Ar-%d][Ac-%d][complex][float32]", N, M));

    //cout << "Printing out the hermitian real part \n";
    //display(Shermitian, 64, 4);

    //cout << "Printing out the hermitian imag part \n";
    //display(Shermitianimag, 64, 4);

    //// Now computing the result from the first multiplication (Z*S^H)--> Result 1
    result1 = (float *)malloc(N * N * sizeof(float));
    result1imag = (float *)malloc(N * N * sizeof(float));

    result2 = (float *)malloc(N * N * sizeof(float));
    result2imag = (float *)malloc(N * N * sizeof(float));

    //cout << "Now computing the matrix multiplication of Z*S^H ---> C3 \n ";

    //cout << "Z:\n";
    //display(Z, 4, 64);
    //cout << "Zi:\n";
    //display(Zi, 4, 64);
    //cout << "Shermitian:\n";
    //display(Shermitian, 64, 4);
    //cout << "Shermitianimag\n";
    //display(Shermitianimag, 64, 4);

    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
    //mmult(Z, Zi, Shermitian, Shermitianimag, result1, result1imag);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res1 = 0; res2 = 0; res3 = 0; res4 = 0;
            for (k = 0; k < M; k++) {
                //float term1 = Abuf[i][k] * Bbuf[k][j];
                term1 = Z[i * M + k] * Shermitian[k * N + j];
                res1 += term1;
                //float term2 = Aibuf[i][k] * Bibuf[k][j];
                term2 = Zi[i * M + k] * Shermitianimag[k * N + j];
                res2 += term2;
                //float term3 = Abuf[i][k] * Bibuf[k][j];
                term3 = Z[i * M + k] * Shermitianimag[k * N + j];
                res3 += term3;
                //float term4 = Aibuf[i][k] * Bbuf[k][j];
                term4 = Zi[i * M + k] * Shermitian[k * N + j];
                res4 += term4;
            }
            result1[i * N + j] = res1 - res2;
            result1imag[i * N + j] = res3 + res4;
        }
    }
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));

    //cout << "Finished first mmult\n";
    //cout << "Real Part \n";
    //display(result1, 4, 4);

    //cout << "Imag Part \n";
    //display(result1imag, 4, 4);

    // Now computing the second matrix multiplication (S*S^H) ---> Result2

    // cout<<"Now computing the result of S*S^H ---> C1 \n";
    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
    //mmult(S, Si, Shermitian, Shermitianimag, result2, result2imag);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res1 = 0; res2 = 0; res3 = 0; res4 = 0;
            for (k = 0; k < M; k++) {
                //float term1 = Abuf[i][k] * Bbuf[k][j];
                term1 = S[i * M + k] * Shermitian[k * N + j];
                res1 += term1;
                //float term2 = Aibuf[i][k] * Bibuf[k][j];
                term2 = Si[i * M + k] * Shermitianimag[k * N + j];
                res2 += term2;
                //float term3 = Abuf[i][k] * Bibuf[k][j];
                term3 = S[i * M + k] * Shermitianimag[k * N + j];
                res3 += term3;
                //float term4 = Aibuf[i][k] * Bbuf[k][j];
                term4 = Si[i * M + k] * Shermitian[k * N + j];
                res4 += term4;
            }
            result2[i * N + j] = res1 - res2;
            result2imag[i * N + j] = res3 + res4;
        }
    }
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));

    //printf("Inside temporal mitigation, I have finished both matrix multiplies!\n");

    //cout << "Real Part \n";
    //display(result2, 4, 4);

    //cout << "Imag part \n";
    //display(result2imag, 4, 4);

    // Now computing the inverse of the above result which is (S*S^H)^-1 ---> D and Di
    // To store inverse of A[][]
    
    inv1 = (float *)malloc(N * N * sizeof(float));
    inv2 = (float *)malloc(N * N * sizeof(float));
    intmedt1 = (float *)malloc(N * N * sizeof(float));
    intmedt2 = (float *)malloc(N * N * sizeof(float));
    intmedb1 = (float *)malloc(N * N * sizeof(float));
    intmedb2 = (float *)malloc(N * N * sizeof(float));
    intmedb3 = (float *)malloc(N * N * sizeof(float));
    intmedb4 = (float *)malloc(N * N * sizeof(float));
    buffer1 = (float *)malloc(N * N * sizeof(float));
    buffer2 = (float *)malloc(N * N * sizeof(float));
    buffer3 = (float *)malloc(N * N * sizeof(float));
    buffer4 = (float *)malloc(N * N * sizeof(float));
    // The following arrays are used for the inverse computation

    bufferinv1 = (float *)malloc(N * N * sizeof(float));
    bufferinv2 = (float *)malloc(N * N * sizeof(float));
    bufferinv3 = (float *)malloc(N * N * sizeof(float));
    bufferinv4 = (float *)malloc(N * N * sizeof(float));
    bufferinv5 = (float *)malloc(N * N * sizeof(float));
    //cout << "Now computing the inverse of the above matrix multiplication ---> C2=(C1)^-1 \n";

    // inverse(result2,inv1);
    // Computing the inverse of a == Real part
    KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
    alternateinverse(result2, inv1);
    KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));

    //display(inv1, 4, 4);
    //cout << "\n";
    // compute res1 =inv(inv1)*result2imag == > bufferinv1

    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
    mmultiply(inv1, result2imag, bufferinv1);
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));

    // compute res2 = inv(result2imag*res1+result2)

    // result2imag*bufferinv1 == > bufferinv2
    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
    mmultiply(result2imag, bufferinv1, bufferinv2);
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));

    // bufferinv2+result2 ==> bufferinv3
    KERN_ENTER(make_label("ZIP[add][%d,%d][complex][float32]",N,N));
    mmadd(bufferinv2, result2, bufferinv3);
    KERN_EXIT(make_label("ZIP[add][%d,%d][complex][float32]",N,N));

    // Now computing inv(bufferinv3) ==> bufferinv4
    // The following 'bufferinv4' is real part of the inverse
    //cout << "\n The imaginary part :::: \n";

    KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
    alternateinverse(bufferinv3, bufferinv4);
    KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));

    //display(bufferinv4, 4, 4);
    // Now computing the imaginary part of the inverse

    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
    mmultiply(bufferinv1, bufferinv4, bufferinv5);
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));

    for (k = 0; k < 4; k++) {
        for (m = 0; m < 4; m++) {
            bufferinv5[k * 4 + m] *= -1;
        }
    }
    // Final result is -bufferinv5 == > imag part of inverse
    //

    //cout << "*****Check for line 162 ********\n ";

    //cout << "Printing the inverse of (S*S^H) C2 --> \n ";

    // Real part of the result involving the inverse
    //cout << "\n Fine uptill here too \n";
    // Currently only having trouble with imaginary part of inverse
    //	imagpart(result2,result2imag,inv1,inv2,intmedt2);
    //cout << "\n";
    //	display(intmedt2,4,4);
    // Now computing the result of (Z*S^H)*(S.S^H)^-1  ---> result3 which is a 4*4 and 4*4 multiplication

    result3 = (float *)malloc(N * N * sizeof(float));
    result3imag = (float *)malloc(N * N * sizeof(float));

    //cout << "Now computing the result (Z*S^H)*(S*S^H)^-1 \n ";

    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
    mmult4(result1, result1imag, bufferinv4, bufferinv5, result3, result3imag);
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));

    //cout << "\n****** Result from line 191 **************\n";

    //cout << "Result from C3*C2 \n";
    //cout << "Real Part \n";
    //display(result3, 4, 4);

    //cout << "Imag Part \n";
    //display(result3imag, 4, 4);

    // Now computing the final matrix multiplication which is result3*S ---> result4 this is 4*4 and 4*64
    // multiplication
    result4 = (float *)malloc(N * M * sizeof(float));
    result4imag = (float *)malloc(N * M * sizeof(float));

    //cout << "Final multiplication is being computed \n";

    KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, M));
    mmult64(result3, result3imag, S, Si, result4, result4imag);
    KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, M));

    //cout << "*******Final Multiplication result Line 221 *********** \n";
    //cout << "Real Part \n";
    //display(result4, 4, 64);

    //cout << "Imag Part \n";
    //display(result4imag, 4, 64);

    // Now we have to compute the final operation which is matrix subtraction : (Z - result4) ---> Zr  4*64 - 4*64
    zres = (float *)malloc(N * M * sizeof(float));
    zresimag = (float *)malloc(N * M * sizeof(float));

    //cout << "Now computing the matrix subtraction from the above result to compute the Z response \n";

    KERN_ENTER(make_label("ZIP[subtract][%d,%d][complex][float32]", N, M));
    msub(Z, Zi, result4, result4imag, zres, zresimag);
    KERN_EXIT(make_label("ZIP[subtract][%d,%d][complex][float32]", N, M));

    //cout << "*****Final result being printed ******\n";
    // Printing the result out
    //cout << "Real part: \n";
    //display(zres, 4, 64);
    //cout << "Imag part: \n";
    //display(zresimag, 4, 64);

    end = clock();

    //cout << "Temporal mitigation complete\n"; // (took " << ((double)(end - begin))/CLOCKS_PER_SEC << " seconds)\n";
    cout << "Temporal mitigation complete\n";

    free(Z);
    free(Zi);
    free(S);
    free(Si);
    free(Shermitian);
    free(Shermitianimag);
    free(result1);
    free(result1imag);
    free(result2);
    free(result2imag);
    free(inv1);
    free(inv2);
    free(bufferinv1);
    free(bufferinv2);
    free(bufferinv3);
    free(bufferinv4);
    free(bufferinv5);
    free(intmedt1);
    free(intmedt2);
    free(result3);
    free(result3imag);
    free(result4);
    free(result4imag);
    free(zres);
    free(zresimag);

    return 0;
}
