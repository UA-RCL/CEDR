// #include <mex.h>
#include <math.h>
// #include <matrix.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include "gsl_fft_mex.c"
#include "gsl_ifft_mex.c"
#include "lfm_gen.c"

/*const float complex I;*/

void xcorr (double *,double *,size_t, double *);

double radar_Rx (double *, double *, double , double , double , double *, size_t);


void xcorr( double *x, double *y, size_t n_samp, double *corr)
{

	size_t len = 2*n_samp - 1;

	double *c = malloc( 2*len *sizeof(double));
	double *d = malloc( 2*len *sizeof(double));

	size_t x_count = 0;
	size_t y_count = 0;

	for(size_t i =0;i<2*len;i+=2)
	{
		if(i/2 > n_samp - 1)
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

		if(i > n_samp)
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

	gsl_ifft(corr_freq,corr,len);

}

double radar_Rx (double *received_signal, double *time, double B, double T, double samp_rate, double *corr, size_t n_samp)
{
	double *gen_wave = malloc( 2*n_samp * sizeof(double));	
	waveform_gen (time,B,T,gen_wave,n_samp);
	double lag;
	//Add code for zero-padding, to make sure signals are of same length
	xcorr(gen_wave, received_signal,n_samp, corr);

	//Code to find maximum
	double max_corr = 0;
	double index = 0;
	for(size_t i =0;i<2*(2*n_samp - 1);i+=2)
	{
		// Only finding maximum of real part of correlation
		if (corr[i] > max_corr)
		{
			max_corr = corr[i];
			index = i/2;
		}

	}

	lag = (n_samp - index)/samp_rate;
	return lag;
}

int main(int argc, char *argv[])
{
		//MAIN FUNCTION TO ACCEPT INPUTS FROM FILE

	//order of arguments: number of samples,B,T,sampling_rate
	size_t n_samples = atoi(argv[1]);
	double T = atof(argv[2]);
	double B = atof(argv[3]);
	double sampling_rate = atof(argv[4]);

	double *time = malloc(n_samples*sizeof(double));;
	double *received = malloc(2*n_samples*sizeof(double));
	
	FILE *fp;
	fp = fopen("/home/alex/Documents/MATLAB/DASH DSoC Code/time_input.txt","r");

	for(size_t i=0; i<n_samples; i++) 
	{
		fscanf(fp,	"%lf", &time[i]);
	}	
	fclose(fp);

	fp = fopen("/home/alex/Documents/MATLAB/DASH DSoC Code/received_input.txt","r");	
	
	for(size_t i=0; i<2*n_samples; i++) 
	{
		fscanf(fp,"%lf", &received[i]);
	}	
	fclose(fp);

	double lag;
	double *corr = malloc( (2*(2*n_samples - 1)) * sizeof(double));

	lag = radar_Rx (received, time, B, T,sampling_rate, corr, n_samples);

	fp = fopen("/home/alex/Documents/MATLAB/DASH DSoC Code/lag_output.txt","w");	
	fprintf(fp,"Lag Value is: %lf",lag);
	fclose(fp);
}




// int main(int argc, char *argv[])
// {
// 		//MAIN FUNCTION TO ACCEPT INPUTS FROMM TERMINAL

// 	//order of arguments: number of samples,B,T,sampling_rate
// 	size_t n_samples = atoi(argv[1]);
// 	double T = atof(argv[2]);
// 	double B = atof(argv[3]);
// 	double sampling_rate = atof(argv[4]);

// 	double *time = malloc(n_samples*sizeof(double));;
// 	double *received = malloc(2*n_samples*sizeof(double));
	
// 	printf("Enter %zu elements in time : ", n_samples);
// 	for(size_t i=0; i<n_samples; i++) 
// 	{
// 		scanf("%lf", &time[i]);
// 	}	
// 	printf("Enter %zu elements in the received signal : ", 2*n_samples);
// 	for(size_t i=0; i<2*n_samples; i++) 
// 	{
// 		scanf("%lf", &received[i]);
// 	}	

// 	double lag;
// 	double *corr = malloc( (2*(2*n_samples - 1)) * sizeof(double));

// 	lag = radar_Rx (received, time, B, T,sampling_rate, corr, n_samples);
// 	printf("Lag Value is: %lf",lag);
// }


// void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// 	/* input variable declaration */

// 	// Order of function call: received_vector,time, B,T,sampling rate

// 	mxComplexDouble *mx_received;
// 	mxDouble *mx_time;
// 	double T;
// 	double B;
// 	double sampling_rate;
// 	size_t n_samples;


// 	/* output variable declaration */
// 	mxDouble *mx_lag;
// 	mxComplexDouble *mx_corr;

// 	/* make these C variables point to the matlab workspace data */
// 	mx_received =  mxGetComplexDoubles(prhs[0]);
// 	mx_time = mxGetDoubles(prhs[1]);

// 	/* find out how many data points there are on the input array */
// 	n_samples = mxGetNumberOfElements(prhs[0]);

// 	B = mxGetScalar(prhs[2]);

// 	T = mxGetScalar(prhs[3]);

// 	sampling_rate = mxGetScalar(prhs[4]);

// 	/* allocate memory for the output data, same size as the input array */
// 	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
// 	plhs[1] = mxCreateDoubleMatrix((2*n_samples - 1), 1, mxCOMPLEX);

// 	/* make our C output variable point to this matlab workspace data */
// 	mx_lag = mxGetDoubles(plhs[0]);
// 	mx_corr = mxGetComplexDoubles(plhs[1]); 

// 	// allocate normal interleaved C arrays
// 	double *time = malloc(n_samples*sizeof(double));;
// 	double *received = malloc(2*n_samples*sizeof(double));
// 	double lag;
// 	double *corr = malloc( (2*(2*n_samples - 1)) * sizeof(double));

// 		// copy the input data to input_array
// 	for(size_t i=0; i<2*n_samples; i+=2) 
// 	{
// 		time [i/2] = mx_time[i/2];
// 		received[i]   = mx_received[i/2].real;
// 		received[i+1] = mx_received[i/2].imag;
// 	}


// 	/* do the calculation with the C code */
// 	lag = radar_Rx (received, time, B, T,sampling_rate, corr, n_samples);

// 	// copy the output_array to the matlab workspace output	
// 	*mx_lag = lag;

// 	for(size_t i=0; i<2*n_samples; i+=2) 
// 	{
// 		mx_corr[i/2].real = corr[i];
// 		mx_corr[i/2].imag = corr[i+1];
// 	}

// }