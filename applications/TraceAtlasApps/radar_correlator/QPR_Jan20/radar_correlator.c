// #include <mex.h>
//#include <math.h>
// #include <matrix.h>
//#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// "/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/radar_correlator"

int main(void)
{
    size_t len, x_count, y_count, i;
    size_t n_samples = 256;
    size_t dft_size = 2 * n_samples-1;

    double lag;
    double T = 0.000512;
    double B = 500000;
    double sampling_rate = 1000;
    double max_corr = 0;
    double index = 0;

    double *c, *d, *X1, *X2, *corr_freq;
    double *time, *received, *dftMatrix, *indftMatrix, *corr, *gen_wave;
    double *dummy_array = malloc(10 * sizeof(double));
    clock_t begin, end;
    clock_t dft_begin1, dft_end1, dft_begin2, dft_end2;

    FILE *fp;

    int row, column;

    // Force the following variable assignments into not-the-first basic block where all the AllocaInsts occur.
    for (i = 0; i < 10; i++) {
        dummy_array[i] = i * 1.1f;
    }
   
    begin = clock(); 
    time = malloc(n_samples*sizeof(double));
    received = malloc(2*n_samples*sizeof(double));
    dftMatrix = malloc(2* dft_size * dft_size * sizeof(double));
    indftMatrix = malloc(2 * dft_size * dft_size * sizeof(double));
    corr = malloc( (2*(2*n_samples - 1)) * sizeof(double));
    gen_wave = malloc(2 * n_samples * sizeof(double));

    fp = fopen("./input/time_input.txt","r");
    //printf("Reading time input");
    for(i = 0; i < n_samples; i++)
    {
        //printf(".");
            fscanf(fp,	"%lf", &time[i]);
    }
    //printf("\n");
    fclose(fp);

    //printf("Generating original wave");
    for (i = 0; i < 2 * n_samples; i += 2)
    {
        //printf(".");
        gen_wave[i] = sin(M_PI * B / T * pow(time[i / 2], 2));
        gen_wave[i + 1] = cos(M_PI * B / T * pow(time[i / 2], 2));
    }
    //printf("\n");


    fp = fopen("./input/received_input.txt","r");
    //printf("Reading received file");
    for(i=0; i<2*n_samples; i++)
    {
        //printf(".");
            fscanf(fp,"%lf", &received[i]);
    }
    //printf("\n");
    fclose(fp);

    fp = fopen("./input/dftcoe.txt", "r");
    //printf("Reading DFT Matrix");
    for (i = 0; i<dft_size * dft_size *2; i++)
    {
        //printf(".");
            fscanf(fp, "%lf", &dftMatrix[i]);
    }
    //printf("\n");
    fclose(fp);

    fp = fopen("./input/indftcoe.txt", "r");
    //printf("Reading IDFT Matrix");
    for (i = 0; i<dft_size * dft_size * 2; i++)
    {
        //printf(".");
            fscanf(fp, "%lf", &indftMatrix[i]);
    }
    //printf("\n");
    fclose(fp);

    //Add code for zero-padding, to make sure signals are of same length

    len = 2 * n_samples - 1;

    c = malloc(2 * len * sizeof(double));
    d = malloc(2 * len * sizeof(double));

    x_count = 0;
    y_count = 0;

    //printf("Zero padding the received wave");
    for (i = 0; i < 2 * len; i += 2)
    {
        //printf(".");
            (i / 2 > n_samples - 1) && ((c[i] = gen_wave[x_count], c[i + 1] = gen_wave[x_count + 1], x_count += 2) || 1) || (c[i] = 0, c[i + 1] = 0);

            (i > n_samples) && ((d[i] = 0, d[i + 1] = 0) || 1) || (d[i] = received[y_count], d[i + 1] = received[y_count + 1], y_count += 2);

    }
    //printf("\n");

    X1 = malloc(2 * len * sizeof(double));
    X2 = malloc(2 * len * sizeof(double));
    corr_freq = malloc(2 * len * sizeof(double));
    //printf("Performing DFT");
    dft_begin1 = clock();
    for (i = 0; i < dft_size * dft_size *2; i += 2)
    {
        //printf(".");
            row = i /512;
            column = i % 512;
            X1[2*row] += dftMatrix[i] * c[column];
            X1[2*row+1] += dftMatrix[i+1] * c[column+1];
    }
    dft_end1 = clock();
    printf("DFT 1 Time: %lf\n", (double)(dft_end1-dft_begin1)/CLOCKS_PER_SEC);
    //printf("\n");

    //printf("Performing DFT2");
    dft_begin2 = clock();
    for (i = 0; i < dft_size * dft_size * 2; i += 2)
    {
        //printf(".");
            row = i / 512;
            column = i % 512;
            X2[2 * row] += dftMatrix[i] * d[column];
            X2[2 * row + 1] += dftMatrix[i + 1] * d[column + 1];
    }
    dft_end2 = clock();
    printf("DFT 2 Time: %lf\n", (double)(dft_end2-dft_begin2)/CLOCKS_PER_SEC);
    //printf("\n");

    //printf("Calculating correlation between the two signals");
    for (i = 0; i < 2 * len; i += 2)
    {
        //printf(".");
            corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
            corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
    }
    //printf("\n");


    //printf("Calculating IDFT");
    for (i = 0; i < dft_size * dft_size * 2; i += 2)
    {
        //printf(".");
            row = i / 512;
            column = i % 512;
            column += 1;
            column -= 1;
            corr[2 * row] += indftMatrix[i] * corr_freq[column];
            corr[2 * row + 1] += indftMatrix[i + 1] * corr_freq[column + 1];
    }
    //printf("\n");

    //Code to find maximum
    //printf("Finding index of maximum");
    for (i = 0; i < 2 * (2 * n_samples - 1); i += 2)
    {
        //printf(".");
            // Only finding maximum of real part of correlation
            (corr[i] > max_corr) && (max_corr = corr[i], index = i / 2);

    }
    //printf("\n");
    lag = (n_samples - index) / sampling_rate;
    end = clock();
    printf("Lag Value is: %lf (time taken: %lf)\n", lag, (double)(end-begin)/CLOCKS_PER_SEC);
}

