// #include "mex.h"
// #include "Matrix.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

// mex -lgsl -lgslcblas -lm -R2018a gsl_fft_mex.c
void gsl_fft(double *, double *, size_t);

void gsl_fft(double *input_array, double *output_array, size_t n_elements) {
	// allocate wavetable and workspace for this length FFT
	gsl_fft_complex_wavetable *wavetable = gsl_fft_complex_wavetable_alloc(n_elements);
	gsl_fft_complex_workspace *workspace = gsl_fft_complex_workspace_alloc(n_elements);
	
	// since this is an in-place transform, copy the input data to the output array
	for(size_t i=0; i<2*n_elements; i++) {
		output_array[i] = input_array[i];
	}
	
	// do the transform
	gsl_fft_complex_forward(output_array, 1, n_elements, wavetable, workspace);
	
	// deallocate wavetable and workspace
	gsl_fft_complex_wavetable_free(wavetable);
	gsl_fft_complex_workspace_free(workspace);
}

/*void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// get a pointer to the matlab workspace data
	mxComplexDouble *mx_input_array = mxGetComplexDoubles(prhs[0]);

	// find out how many data points there are on the input array
	size_t n_elements = mxGetNumberOfElements(prhs[0]);

	// allocate memory for the output data, same size as the input array
	plhs[0] = mxCreateDoubleMatrix(n_elements, 1, mxCOMPLEX);

	// make our C output variable point to this matlab workspace data
	mxComplexDouble *mx_output_array = mxGetComplexDoubles(plhs[0]);

	// allocate normal interleaved C arrays
	double input_array[2*n_elements];
	double output_array[2*n_elements];

	// copy the input data to input_array
	for(size_t i=0; i<2*n_elements; i+=2) {
		input_array[i]   = mx_input_array[i/2].real;
		input_array[i+1] = mx_input_array[i/2].imag;
	}

	// do the calculation with the C code
	gsl_fft(input_array, output_array, n_elements);

	// copy the output_array to the matlab workspace output
	for(size_t i=0; i<2*n_elements; i+=2) {
		mx_output_array[i/2].real = output_array[i];
		mx_output_array[i/2].imag = output_array[i+1];
	}
}*/