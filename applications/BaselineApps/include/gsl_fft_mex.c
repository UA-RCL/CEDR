#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

void gsl_fft(double *, double *, size_t);

void gsl_fft(double *input_array, double *output_array, size_t n_elements) {
	// allocate wavetable and workspace for this length FFT
	gsl_fft_complex_wavetable *wavetable = gsl_fft_complex_wavetable_alloc(n_elements);
	gsl_fft_complex_workspace *workspace = gsl_fft_complex_workspace_alloc(n_elements);

	// since this is an in-place transform, copy the input data to the output array
	for (size_t i = 0; i < 2 * n_elements; i++) {
		output_array[i] = input_array[i];
	}

	// do the transform
	gsl_fft_complex_forward(output_array, 1, n_elements, wavetable, workspace);

	// deallocate wavetable and workspace
	gsl_fft_complex_wavetable_free(wavetable);
	gsl_fft_complex_workspace_free(workspace);
}
