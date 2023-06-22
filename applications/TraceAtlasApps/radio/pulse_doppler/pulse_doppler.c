#include <stdio.h>
//#include <mex.h>
#include <math.h>
//#include <matrix.h>
#include <stdlib.h>
//#include <complex.h>
//#include <time.h>

#include <sys/time.h>

/* Function Declarations */
void xcorr (double *,double *,size_t, size_t, double *);
void swap(double *, double *);
//void fftshift(double *, double );
void fftshift(double *, size_t);
void gsl_fft(double *, double *, size_t);
void gsl_ifft(double *, double *, size_t);

//int main(int argc, char *argv[])
int main(void)
{
    //size_t m = atoi(argv[1]);                               // number of pulses
    //size_t n_samples = atoi(argv[2]);                       // length of single pulse
    size_t m = 128;
    size_t n_samples = 64;
    int i, j, k, n, x, y, z;
    //clock_t begin, end;

    struct timeval tv1, tv2;

    double **mf;// = malloc(2*n_samples * sizeof(double*));
    //double mf[2*n_samples-1][m*2]; // build a 2D array for the output of the matched filter
    double *p;// = malloc(2*n_samples *sizeof(double));   // array for pulse with noise
    double *pulse;// = malloc(2*n_samples *sizeof(double));  // array for the original pulse
    double *corr;// = malloc(2*(2*n_samples - 1) *sizeof(double));   // array for the output of matched filter
    /* create arrays for FFT */
    
    //double q[2*m], r[m], f[m][2*n_samples-1];
    double *q;
    double *r;
    double **f;

    FILE *fp;

    for (i = 0; i < 1; i++) {}

    //begin = clock();
    gettimeofday(&tv1, NULL);

    mf = malloc(2*n_samples*sizeof(double*));
    p = malloc(2*n_samples*sizeof(double));
    pulse = malloc(2*n_samples*sizeof(double));
    corr = malloc(2*(2*n_samples-1)*sizeof(double));

    q = malloc(2 * m * sizeof(double));
    r = malloc(m * sizeof(double));
    f = malloc(m * sizeof(double*));

    for (i = 0; i < 2*n_samples; i++) {
        mf[i] = malloc(m*2*sizeof(double));
    }
    for (i = 0; i < m; i++) {
        f[i] = malloc((2*n_samples-1) * sizeof(double));
    }

    fp = fopen("./input/input_pd_pulse.txt","r");  // read the original pulse
    for(i=0; i<2*n_samples; i++) 
	{
		fscanf(fp, "%lf", &pulse[i]);
	}
    fclose(fp);
    
    /* matched filter */
    
    fp = fopen("./input/input_pd_ps.txt","r");   // read the multiple pulses with noise and delay
    for(k = 0; k < m; k++)
    {        
        for(j = 0; j < 2 * n_samples; j++)
        {
            fscanf(fp, "%lf", &p[j]);
        }
        
        /* matched filter */
        
        xcorr(p, pulse, n_samples, n_samples, corr);
        
        /* put the output into a new 2D array */
        
        for(n = 0; n < 2*(2 * n_samples - 1); n+=2)
        {
            mf[n/2][2*k] = corr[n];
            mf[n/2][2*k+1] = corr[n+1];
        }
    }
    fclose(fp);
    
    /* FFT */
    
    for(x = 0; x < 2*n_samples-1; x++)
    {
        gsl_fft(mf[x], q, m);
        for(y = 0; y < 2*m; y+=2)
        {
            r[y/2] = sqrt(q[y]*q[y]+q[y+1]*q[y+1]);   // calculate the absolute value of the output 
        }
        fftshift(r, m);
        
        for(z = 0; z < m; z++)
        {
            f[z][x] = r[z];      // put the elements of output into corresponding location of the 2D array
        }
    }
//    fp = fopen("/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/radio/pulse_doppler/output_pd_f.txt","w");  // write the output
//    for(i = 0; i < m; i++)
//    {
//        for(j = 0; j < 2*n_samples-1; j++)
//        {
//            fprintf(fp, "%lf ", f[i][j]);
//        }
//        fprintf(fp, "\n");
//    }
//    fclose(fp);
    gettimeofday(&tv2, NULL);
    //end = clock();
    printf("Pulse doppler complete (took: %f seconds)\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec)); //((double)(end - begin))/CLOCKS_PER_SEC);
}

/*
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* input and output variables declaration */
    
/*    mxComplexDouble *mx_ps;                 // define the original pulse without noise, delay and Doppler frequency shift
    mxComplexDouble *mx_pulse;              // define pulses with noise
    mxDouble *mx_f;                         // define output
	double m;                               // number of pulses
	size_t n_samples;                       // length of single pulse
    int i, j, k, n, x, y, z;
    
    /* make the input C variables point to the matlab workspace data */

/*    mx_ps = mxGetComplexDoubles(prhs[0]);   
    mx_pulse = mxGetComplexDoubles(prhs[1]);
    
    /* find out the size of the input arrays */
 
/*    m = mxGetM(prhs[0]);   // find the number of pulses
    n_samples = mxGetNumberOfElements(prhs[1]); // find the length of a single pulse
    
    double mf[2*n_samples-1][(int)m*2]; // build a 2D array for the output of the matched filter
    
    double p[2*n_samples];   // array for pulse with noise
    double pulse[2*n_samples];  // array for the original pulse
    double corr[2*(2*n_samples - 1)];   // array for the output of matched filter
    
    /* separate the real and imaginary part of the original pulse and put it into a new array */
    
/*    for(i=0; i<2*n_samples; i+=2) 
	{
		pulse[i]   = mx_pulse[i/2].real;   
		pulse[i+1] = mx_pulse[i/2].imag;
	}
    
    /* matched filter */
    
/*    for(k = 0; k < m; k++)
    {
        /* separate the real and imaginary part of each pulse and put it into a new array */
        
/*        for(j = 0; j < 2 * n_samples; j+=2)
        {
            p[j] = mx_ps[k+j/2*(int)m].real;
            p[j+1] = mx_ps[k+j/2*(int)m].imag;
        }
        
        /* matched filter */
        
/*        xcorr(p, pulse, n_samples, n_samples, corr);
        
        /* put the output into a new 2D array */
        
/*        for(n = 0; n < 2*(2 * n_samples - 1); n+=2)
        {
            mf[n/2][2*k] = corr[n];
            mf[n/2][2*k+1] = corr[n+1];
        }
    }
    
    /* make the output C variable point to the matlab workspace data */

/*    plhs[0] = mxCreateDoubleMatrix(m, (2*n_samples-1), mxREAL);
    mx_f = mxGetDoubles(plhs[0]);
    
    /* create arrays for FFT */
    
/*    double q[2*(int)m], r[(int)m];
    
    /* FFT */
    
/*    for(x = 0; x < 2*n_samples-1; x++)
    {
        gsl_fft(mf[x], q, m);
        for(y = 0; y < 2*m; y+=2)
        {
            r[y/2] = sqrt(q[y]*q[y]+q[y+1]*q[y+1]);   // calculate the absolute value of the output 
        }
        fftshift(r, m);
        for(z = 0; z < m; z++)
        {
            mx_f[z+(int)m*x] = r[z];      // put the elements of output into corresponding location of the output matrix
        }
    }
}*/

#include "gsl_fft_mex.c"
#include "gsl_ifft_mex.c"

/* Function Definitions */
// __attribute__((always_inline))
void xcorr( double *x, double *y, size_t n_samp1, size_t n_samp2, double *corr)
{
    size_t len;
    if (n_samp1 > n_samp2)
    {
        len = 2*n_samp1 - 1;
    }
    else
    {
        len = 2*n_samp2 - 1;
    }

    double *c = malloc( 2*len *sizeof(double));
    double *d = malloc( 2*len *sizeof(double));

    if (n_samp1 < n_samp2)
    {
        size_t x_count = 0;
        size_t y_count = 0;

        for(size_t i =0;i<2*len;i+=2)
        {
            if( (i/2 > (n_samp2 - 1)) && (i/2 < (n_samp2 + n_samp1 - 1)) )
            {
                c[i] = x[x_count];
                c[i+1] = x[x_count+1];
                x_count+=2;
            }
            else
            {
                c[i] = 0;
                c[i+1] = 0;
            }
            if(i > n_samp2)
            {
                d[i] = 0;
                d[i+1] = 0;
            }
            else
            {
                d[i] = y[y_count];
                d[i+1] = y[y_count+1];
                y_count+=2;
            }
        }
    }
    else
    {
        size_t x_count = 0;
        size_t y_count = 0;

        for(size_t i =0;i<2*len;i+=2)
        {
            if(i/2 > (n_samp1 - 1))
            {
                c[i] = x[x_count];
                c[i+1] = x[x_count+1];
                x_count+=2;
            }
            else
            {
                c[i] = 0;
                c[i+1] = 0;
            }
            if(i > n_samp2)
            {
                d[i] = 0;
                d[i+1] = 0;
            }
            else
            {
                d[i] = y[y_count];
                d[i+1] = y[y_count+1];
                y_count+=2;
            }
        }
    }

    double *X1 = malloc(2*len *sizeof(double));
    double *X2 = malloc(2*len *sizeof(double));
    double *corr_freq = malloc(2*len *sizeof(double));
    gsl_fft(c, X1, len);
    gsl_fft(d, X2, len);

    for(size_t i =0;i<2*len;i+=2)
    {
        corr_freq[i] = (X1[i] * X2[i]) + (X1[i+1] * X2[i+1]);
        corr_freq[i+1] = (X1[i+1] * X2[i]) - (X1[i] * X2[i+1]);
    }

    gsl_ifft(corr_freq, corr, len);

}

// __attribute__((always_inline))
void swap(double *v1, double *v2)
{
    double tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

// __attribute__((always_inline))
void fftshift(double *data, size_t count)
//void fftshift(double *data, double count)
{
    int k = 0;
    //int c = (int) floor((float)count/2);
    int c = count/2;
    // For odd and for even numbers of element use different algorithm
    if (count % 2 == 0)
    {
        for (k = 0; k < c; k++)
        {
            swap(&data[k], &data[k + c]);
        }
    }
    else
    {
        double tmp = data[0];
        for (k = 0; k < c; k++)
        {
            data[k] = data[c + k + 1];
            data[c + k + 1] = data[k + 1];
        }
        data[c] = tmp;
    }
}