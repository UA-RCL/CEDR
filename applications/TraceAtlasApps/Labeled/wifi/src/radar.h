struct args_lfm {
    
    size_t time_n_samples;
    size_t n_samples;
    double *time;
    double *lfm_waveform;
    float *lfm_waveform_float;
    double T;
    double B;
};

struct args_lag_detection {

    size_t n_samples;
    float *corr;
    float *max_corr;
    float *index;
    float *lag;
    double sampling_rate;
};

struct args_fftwf_fft {
    
    float *input_array;
    fftwf_complex *in;
    fftwf_complex *out;
    float *output_array;
    size_t n_elements;
    fftwf_plan p;
};

struct args_conjugate {
    float *in1;
    float *in2;
    float *out;
    size_t len;
};

struct args_amplitude {
    size_t m;
    float *q;
    float *r;
};

struct args_fftshift {
    float *data;
    size_t count;
};
