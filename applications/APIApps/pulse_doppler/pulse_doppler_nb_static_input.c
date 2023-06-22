#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>                       
#include "dash.h"
#include "static_inputs.h"
                                          
#define PROGPATH "./input/"               
#define PDPULSE PROGPATH "input_pd_pulse.txt"
#define PDPS PROGPATH "input_pd_ps.txt"   
#define OUTPUT PROGPATH "output_pd_f.txt" 
                                          
#define KERN_ENTER(str)
#define KERN_EXIT(str) 
    
/* Function Declarations */               
void xcorr(double *, double *, size_t, double *);
void swap(double *, double *);  
void fftshift(double *, double);
                       
/* Function Definitions */                
void xcorr(double *x, double *y, size_t n_samp, double *corr) {
  size_t len;    
  len = 2 * n_samp;  
  double *c = malloc(2 * len * sizeof(double));
  double *d = malloc(2 * len * sizeof(double));
                                          
  size_t x_count = 0;
  size_t y_count = 0;

  double *z = malloc(2 * (n_samp) * sizeof(double));    
  for (size_t i = 0; i < 2 * (n_samp); i += 2) { 
    z[i] = 0;    
    z[i + 1] = 0;
  }                  
  for (size_t i = 0; i < 2 * (n_samp - 1); i += 2) {                  
    c[i] = 0;    
    c[i + 1] = 0;
  }                                       
  memcpy(c + 2 * (n_samp - 1), x, 2 * n_samp * sizeof(double));   
  c[2 * len - 2] = 0;
  c[2 * len - 1] = 0;
  memcpy(d, y, 2 * n_samp * sizeof(double));   
  memcpy(d + 2 * n_samp, z, 2 * (n_samp) * sizeof(double)); 
  double *X1 = malloc(2 * len * sizeof(double));        
  double *X2 = malloc(2 * len * sizeof(double));        
  double *corr_freq = malloc(2 * len * sizeof(double));
 
  // gsl_fft(c, X1, len) and gsl_fft(d, X2, len) (independent);   
  // lol scope
  {

    dash_cmplx_flt_type *fft_inp_x1 = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    dash_cmplx_flt_type *fft_out_x1 = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    dash_cmplx_flt_type *fft_inp_x2 = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    dash_cmplx_flt_type *fft_out_x2 = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    bool is_fwd = true;

    for (size_t i = 0; i < len; i++) {
      fft_inp_x1[i].re = (dash_re_flt_type) c[2*i];
      fft_inp_x1[i].im = (dash_re_flt_type) c[2*i+1];
      fft_inp_x2[i].re = (dash_re_flt_type) d[2*i];
      fft_inp_x2[i].im = (dash_re_flt_type) d[2*i+1];
    }


    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    // DASH_FFT_flt(fft_inp, fft_out, len, true /* is_forward_transform? */);
    DASH_FFT_flt_nb(&fft_inp_x1, &fft_out_x1, &len, &is_fwd, &barrier);
    DASH_FFT_flt_nb(&fft_inp_x2, &fft_out_x2, &len, &is_fwd, &barrier);

    while (completion_ctr != 2) {
      pthread_cond_wait(barrier.cond, barrier.mutex);
      //printf("%u xcorr FFTs have been completed...\n", completion_ctr);
    }

    for (size_t i = 0; i < len; i++) {
      X1[2*i] = (double) fft_out_x1[i].re;
      X1[2*i+1] = (double) fft_out_x1[i].im;
      X2[2*i] = (double) fft_out_x2[i].re;
      X2[2*i+1] = (double) fft_out_x2[i].im;
    }

    free(fft_inp_x1);
    free(fft_out_x1);
    free(fft_inp_x2);
    free(fft_out_x2);
  }
  
  // gsl_fft(d, X2, len);
  // lol scope
  // {
  //   dash_cmplx_flt_type *fft_inp = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
  //   dash_cmplx_flt_type *fft_out = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));

  //   for (size_t i = 0; i < len; i++) {
  //     fft_inp[i].re = (dash_re_flt_type) d[2*i];
  //     fft_inp[i].im = (dash_re_flt_type) d[2*i+1];
  //   }

  //   DASH_FFT_flt(fft_inp, fft_out, len, true /* is_forward_transform? */);

  //   for (size_t i = 0; i < len; i++) {
  //     X2[2*i] = (double) fft_out[i].re;
  //     X2[2*i+1] = (double) fft_out[i].im;
  //   }

  //   free(fft_inp);
  //   free(fft_out);
  // }

  free(c);
  free(d);
  free(z);
  KERN_ENTER(make_label("ZIP[multiply][complex][float64][%d]", len));
  for (size_t i = 0; i < 2 * len; i += 2) {
    corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
    corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
  }
  KERN_EXIT(make_label("ZIP[multiply][complex][float64][%d]", len));
  free(X1);
  free(X2);

  // gsl_ifft(corr_freq, corr, len);
  // lol scope
  {
    dash_cmplx_flt_type *fft_inp = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    dash_cmplx_flt_type *fft_out = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    bool is_fwd = false;
    for (size_t i = 0; i < len; i++) {
      fft_inp[i].re = (dash_re_flt_type) corr_freq[2*i];
      fft_inp[i].im = (dash_re_flt_type) corr_freq[2*i+1];
    }

    // DASH_FFT_flt(fft_inp, fft_out, len, false /* is_forward_transform? */);
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    DASH_FFT_flt_nb(&fft_inp, &fft_out, &len, &is_fwd, &barrier);

    while (completion_ctr != 1) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
    //printf("%u IFFTs have been completed...\n", completion_ctr);
  }

    for (size_t i = 0; i < len; i++) {
      corr[2*i] = (double) fft_out[i].re;
      corr[2*i+1] = (double) fft_out[i].im;
    }

    free(fft_inp);
    free(fft_out);
  }

  free(corr_freq);
}

void swap(double *v1, double *v2) {
  double tmp = *v1;
  *v1 = *v2;
  *v2 = tmp;
}

void fftshift(double *data, double count) {
  int k = 0;
  int c = (double)floor((float)count / 2);
  // For odd and for even numbers of element use different algorithm
  if ((int)count % 2 == 0) {
    for (k = 0; k < c; k++)
      swap(&data[k], &data[k + c]);
  } else {
    double tmp = data[0];
    for (k = 0; k < c; k++) {
      data[k] = data[c + k + 1];
      data[c + k + 1] = data[k + 1];
    }
    data[c] = tmp;
  }
}

int main(int argc, char *argv[]) {
  size_t m = 128;    // number of pulses
  size_t n_samples = 64; // length of single pulse
  double PRI = 1.27e-4;
  int i, j, k, n, x, y, z, o;

  double *mf = malloc((2 * n_samples) * m * 2 *
                      sizeof(double)); // build a 2D array for the output of the
                                       // matched filter
/*
  double *p =
      malloc(2 * n_samples * sizeof(double)); // array for pulse with noise
  double *pulse =
      malloc(2 * n_samples * sizeof(double)); // array for the original pulse
*/
  double *corr =
      malloc(2 * (2 * n_samples) *
             sizeof(double)); // array for the output of matched filter

  // Read the original pulse
  FILE *fp;
/*
  fp = fopen(PDPULSE, "r");
  for (i = 0; i < 2 * n_samples; i++) {
    fscanf(fp, "%lf", &pulse[i]);
  }
  fclose(fp);
*/

  // Run the input samples through the matched filter
//  fp = fopen(PDPS, "r"); // read the multiple pulses with noise and delay
  for (k = 0; k < m; k++) {
/*    for (j = 0; j < 2 * n_samples; j++) {
      fscanf(fp, "%lf", &p[j]);
      if(j == 2 * n_samples-1)
      printf("%lf},\n", p[j]);
      else if (j == 0)
      printf("{%lf, ", p[j]);
      else
      printf("%lf, ", p[j]);
    }
*/

    /* matched filter */
    xcorr(p[k], pulse, n_samples, corr);

    /* put the output into a new 2D array */
    for (n = 0; n < 2 * (2 * n_samples); n += 2) {
      mf[n / 2 + (2 * k) * (2 * n_samples)] = corr[n];
      mf[n / 2 + (2 * k + 1) * (2 * n_samples)] = corr[n + 1];
    }
  }
//  fclose(fp);
//  free(p);
//  free(pulse);
  free(corr);

  double **q = malloc(2*n_samples* sizeof(double));
  double **r = malloc(2*n_samples* sizeof(double));
  double *l = malloc(2 * m * sizeof(double));
  double *f = malloc(m * (2 * n_samples) * sizeof(double));
  double max = 0, a, b;
  

  /* FFT */
  // gsl_fft(l,q,m);
  // lol scope
  {
  int num_ffts = 2*n_samples;
  bool is_fwd = true;
  dash_cmplx_flt_type **fft_inp = (dash_cmplx_flt_type**) malloc(num_ffts * sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type **fft_out = (dash_cmplx_flt_type**) malloc(num_ffts * sizeof(dash_cmplx_flt_type));
  
  for (x = 0; x < num_ffts; x++) {
    
    for (o = 0; o < 2 * m; o++) {
      l[o] = mf[x + o * num_ffts];
    }
    
      fft_inp[x] = (dash_cmplx_flt_type*) malloc(m * sizeof(dash_cmplx_flt_type));
      fft_out[x] = (dash_cmplx_flt_type*) malloc(m * sizeof(dash_cmplx_flt_type));

      for (size_t i = 0; i < m; i++) {
        fft_inp[x][i].re = (dash_re_flt_type) l[2*i];
        fft_inp[x][i].im = (dash_re_flt_type) l[2*i+1];
      }
  }
  // exit for loop to do non-blocking api calls over array of fft_inp
  // DASH_FFT_flt(fft_inp, fft_out, m, true /* is_forward_transform? */);
 
  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (size_t i = 0; i < num_ffts; i++) {
//    printf("calling the %ld-th API\n", i);
    DASH_FFT_flt_nb(&fft_inp[i], &fft_out[i], &m, &is_fwd, &barrier);
  }

  while (completion_ctr != num_ffts) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
//    printf("%u FFTs have been completed for gsl_fft(l,q,m)...\n", completion_ctr);
  }
  pthread_mutex_unlock(barrier.mutex);

  for(x = 0; x < num_ffts; x++){
      q[x] = malloc(2 * m *sizeof(double));
      for (size_t i = 0; i < m; i++) {
        q[x][2*i] = (double) fft_out[x][i].re;
        q[x][2*i+1] = (double) fft_out[x][i].im;
      }
  }

  free(fft_inp);
  free(fft_out);
    
  for(x = 0; x < num_ffts; x++){
    r[x] = malloc(m *sizeof(double));
    for (y = 0; y < 2 * m; y += 2) {
      r[x][y / 2] = sqrt(
          q[x][y] * q[x][y] +
          q[x][y + 1] * q[x][y + 1]); // calculate the absolute value of the output
    }
    fftshift(r[x], m);

    for (z = 0; z < m; z++) {
      f[x + z * num_ffts] = r[x][z]; // put the elements of output into corresponding location of the 2D array
      if (r[x][z] > max) {
        max = r[x][z];
        a = z + 1;
        b = x + 1;
      }
    }
  }
  }
  
  free(mf);
  free(q);
  free(r);
  free(l);

  double rg, dp;
  rg = (b - n_samples) / (n_samples - 1) * PRI;
  dp = (a - (m + 1) / 2) / (m - 1) / PRI;

  fp = fopen("./output/pulse_doppler_output.txt", "w");
  if (fp != NULL) {
    fprintf(fp, "Doppler shift = %lf, time delay = %lf\n", dp, rg);
    fclose(fp);
  }

  //printf("[Pulse Doppler] Execution is complete...\n");
}
