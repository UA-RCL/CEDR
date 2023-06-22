// #include <mex.h>
#include <math.h>
// #include <Matrix.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>


/*const float complex I;*/

void waveform_gen (double *,double, double, double*, size_t);

void waveform_gen (double *time, double B, double T,double *lfm_waveform, size_t n_samples)
{
	
	for (size_t i = 0; i < 2*n_samples; i+=2)
	{
		lfm_waveform[i] = creal(cexp(I *  M_PI * B/T * pow(time[i/2],2)));
		lfm_waveform[i+1] = cimag(cexp(I *  M_PI * B/T * pow(time[i/2],2)));
	}

}


// int main(int argc, char *argv[])
// {
// 		//MAIN FUNCTION TO ACCEPT INPUTS FROM FILE

// 	//order of arguments: number of samples,B,T,sampling_rate
// 	size_t n_samples = atoi(argv[1]);
// 	double T = atof(argv[2]);
// 	double B = atof(argv[3]);

// 	double *time = malloc(n_samples*sizeof(double));;
	
// 	FILE *fp;
// 	fp = fopen("/home/alex/Documents/MATLAB/DASH DSoC Code/time_input.txt","r");

// 	for(size_t i=0; i<n_samples; i++) 
// 	{
// 		fscanf(fp,	"%lf", &time[i]);
// 	}	
// 	fclose(fp);

// 	double *lfm_waveform = malloc(2*n_samples*sizeof(double));

// 	waveform_gen(time, B, T, lfm_waveform, n_samples);

// 	fp = fopen("/home/alex/Documents/MATLAB/DASH DSoC Code/lfm_waveform_output.txt","w");	
// 	for(size_t i=0; i<2*n_samples; i++) 
// 	{
// 			fprintf(fp,"%lf\n",lfm_waveform[i]);
// 	}	
// 	fclose(fp);
// }


// int main(int argc, char *argv[])
// {
// 		//MAIN FUNCTION TO ACCEPT INPUTS FROMM TERMINAL

// 	//order of arguments: number of samples,B,T,sampling_rate
// 	size_t n_samples = atoi(argv[1]);
// 	double T = atof(argv[2]);
// 	double B = atof(argv[3]);

// 	double *time = malloc(n_samples*sizeof(double));;
	
// 	printf("Enter %zu elements in time : ", n_samples);
// 	for(size_t i=0; i<n_samples; i++) 
// 	{
// 		scanf("%lf", &time[i]);
// 	}	
// 	double *lfm_waveform = malloc(2*n_samples*sizeof(double));

// 	waveform_gen(time, B, T, lfm_waveform, n_samples);

// 	for(size_t i=0; i<2*n_samples; i++) 
// 	{
// 			printf("%lf\n",lfm_waveform[i]);
// 	}	
// }

// void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// 	/* input variable declaration */

// 	mxDouble *mx_time;
// 	double T;
// 	double B;
// 	size_t n_samples;

// 	/* output variable declaration */
// 	mxComplexDouble *mx_lfm_waveform;

// 	/* make these C variables point to the matlab workspace data */
// 	mx_time = mxGetDoubles(prhs[0]);

// 	/* find out how many data points there are on the input array */
// 	n_samples = mxGetNumberOfElements(prhs[0]);

// 	B = mxGetScalar(prhs[1]);

// 	T = mxGetScalar(prhs[2]);

// 	/* allocate memory for the output data, same size as the input array */
// 	plhs[0] = mxCreateDoubleMatrix(1, n_samples, mxCOMPLEX);

// 	/* make our C output variable point to this matlab workspace data */
// 	mx_lfm_waveform = mxGetComplexDoubles(plhs[0]);

// 	// allocate normal interleaved C arrays
// 	double *time = malloc(n_samples * sizeod(double));

// 	double lfm_waveform = malloc(2*n_samples *sizeof(double));
// 	for(size_t i = 0;i<n_samples; i++)
// 	{
// 		time [i] = mx_time[i];
// 	}

// 	/* do the calculation with the C code */
// 	waveform_gen(time, B, T, lfm_waveform, n_samples);

// 	// copy the output_array to the matlab workspace output	
// 	for(size_t i = 0;i<2*n_samples; i+=2)
// 	{
// 		mx_lfm_waveform[i/2].real = lfm_waveform[i];
// 		mx_lfm_waveform[i/2].imag = lfm_waveform[i+1];
// 	}
// }