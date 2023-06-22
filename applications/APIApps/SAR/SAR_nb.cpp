#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dash.h"

#define PROGPATH "./input/"
#define RAWDATA PROGPATH "rawdata_rda.txt"
#define OUTPUT "SAR_RDA-output.txt"

// Define KERN_ENTER and KERN_EXIT as NOPs
#define KERN_ENTER(KERN_STR)
#define KERN_EXIT(KERN_STR)

void swap(dash_re_flt_type *, dash_re_flt_type *);
void fftshift(dash_cmplx_flt_type *, double);

void swap(dash_re_flt_type *v1, dash_re_flt_type *v2) {
  dash_re_flt_type tmp = *v1;
  *v1 = *v2;
  *v2 = tmp;
}

void fftshift(dash_cmplx_flt_type *data, double count) {
  int k = 0;
  int c = (double)floor(count / 2);
  // For odd and for even numbers of element use different algorithm
  if ((int)count % 2 == 0) {
    for (k = 0; k < c; k += 1) {
      swap(&data[k].re, &data[k + c].re);
      swap(&data[k].im, &data[k + c].im);
    }
  } else {
    dash_cmplx_flt_type tmp;
    tmp.re = data[0].re;
    tmp.im = data[0].im;
    for (k = 0; k < c; k += 1) {
      data[k].re = data[c + k + 2].re;
      data[k].im = data[c + k + 2].im;
      data[c + k + 2].re = data[k + 2].re;
      data[c + k + 2].im = data[k + 2].im;
    }
    data[c].re = tmp.re;
    data[c].im = tmp.im;
  }
}

int main(void) {
  double c = 3e8;

  int i, j;

  int Nslow;
  int Nfast;
  double v;
  double Xmin;
  double Xmax;
  double Yc;
  double Y0;
  double Tr;
  double Kr;
  double h;
  double lambda;

  double R0;
  double Ka;
  dash_cmplx_flt_type **fft_out_0; // Slow x Fast
  dash_cmplx_flt_type **fft_inp_1; // Slow x Fast
  dash_cmplx_flt_type **fft_out_1; // Slow x Fast
  dash_cmplx_flt_type **fft_inp_2; // Fast x Slow
  dash_cmplx_flt_type **fft_out_2; // Fast x Slow
  dash_cmplx_flt_type **fft_inp_3; // Fast x Slow

  dash_cmplx_flt_type **g; // 1 x Fast
  dash_cmplx_flt_type **g2; // 1 x Fast
  dash_cmplx_flt_type **H; // 1 x Slow

  dash_cmplx_flt_type **s0; // Slow x Fast
  dash_cmplx_flt_type **S1; // Fast x Slow

  FILE *fp;

  double *ta;

  double Rmin, Rmax;
  double *tr;

  // Azimuth Compression
  double *sac;
  
  // printf("[SAR] Starting execution of the non-kernel thread\n");

  Nslow = 256;
  Nfast = 512;
  v = 150;
  Xmin = 0;
  Xmax = 50;
  Yc = 10000;
  Y0 = 500;
  Tr = 2.5e-6;
  Kr = 2e13;
  h = 5000;
  lambda = 0.0566;

  // Allocate Space for all dash arrays
  fft_out_0 = ((dash_cmplx_flt_type**) calloc(Nslow, sizeof(dash_cmplx_flt_type*)));
  fft_inp_1 = ((dash_cmplx_flt_type**) calloc(Nslow, sizeof(dash_cmplx_flt_type*)));
  fft_out_1 = ((dash_cmplx_flt_type**) calloc(Nslow, sizeof(dash_cmplx_flt_type*)));
  fft_inp_2 = ((dash_cmplx_flt_type**) calloc(Nfast, sizeof(dash_cmplx_flt_type*)));
  fft_out_2 = ((dash_cmplx_flt_type**) calloc(Nfast, sizeof(dash_cmplx_flt_type*)));
  fft_inp_3 = ((dash_cmplx_flt_type**) calloc(Nfast, sizeof(dash_cmplx_flt_type*)));

  g = ((dash_cmplx_flt_type**) calloc(1, sizeof(dash_cmplx_flt_type*)));
  g2 = ((dash_cmplx_flt_type**) calloc(1, sizeof(dash_cmplx_flt_type*)));
  H = ((dash_cmplx_flt_type**) calloc(1, sizeof(dash_cmplx_flt_type*)));

  s0 = ((dash_cmplx_flt_type**) calloc(Nslow, sizeof(dash_cmplx_flt_type*)));
  S1 = ((dash_cmplx_flt_type**) calloc(Nfast, sizeof(dash_cmplx_flt_type*)));

  g[0] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
  g2[0] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
  H[0] = (dash_cmplx_flt_type*) calloc(Nslow, sizeof(dash_cmplx_flt_type));

  for(i = 0; i < Nslow; i++){
    fft_out_0[i] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
    fft_inp_1[i] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
    fft_out_1[i] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
    s0[i] = (dash_cmplx_flt_type*) calloc(Nfast, sizeof(dash_cmplx_flt_type));
  }

  for(i = 0; i < Nfast; i++){
    fft_inp_2[i] = (dash_cmplx_flt_type*) calloc(Nslow, sizeof(dash_cmplx_flt_type));
    fft_out_2[i] = (dash_cmplx_flt_type*) calloc(Nslow, sizeof(dash_cmplx_flt_type));
    fft_inp_3[i] = (dash_cmplx_flt_type*) calloc(Nslow, sizeof(dash_cmplx_flt_type));
    S1[i] = (dash_cmplx_flt_type*) calloc(Nslow, sizeof(dash_cmplx_flt_type));
  }

  R0 = sqrt(Yc * Yc + h * h);
  Ka = 2 * v * v / lambda / R0;
  sac = (double*) malloc(Nslow * Nfast * sizeof(double));

  fp = fopen(RAWDATA, "r");
  for (i = 0; i < Nslow; i++) {
    for (j = 0; j < Nfast; j++){
      fscanf(fp, "%f", &s0[i][j].re);
      fscanf(fp, "%f", &s0[i][j].im);
    }
  }
  fclose(fp);

  ta = (double*) malloc(Nslow * sizeof(double));
  ta[0] = 0;
  for (i = 1; i < Nslow; i++) {
    ta[i] = ta[i - 1] + (Xmax - Xmin) / v / (Nslow - 1);
  }

  Rmin = sqrt((Yc - Y0) * (Yc - Y0) + h * h);
  Rmax = sqrt((Yc + Y0) * (Yc + Y0) + h * h);
  tr = (double*) malloc(Nfast * sizeof(double));
  tr[0] = 0;
  for (i = 1; i < Nfast; i++) {
    tr[i] = tr[i - 1] + (2 * Rmax / c + Tr - 2 * Rmin / c) / (Nfast - 1);
  }
  
  //Set up g and H
  for (i = 0; i < Nfast; i += 1) {
    if (tr[i] > -Tr / 2 && tr[i] < Tr / 2) {
      g[0][i].re = cos(M_PI * Kr * tr[i] * tr[i]);
      g[0][i].im = -sin(M_PI * Kr * tr[i] * tr[i]);
    } else {
      g[0][i].re = 0;
      g[0][i].im = 0;
    }
  }

  for (i = 0; i < Nslow; i += 1) {
    if (ta[i] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
        ta[i] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
      H[0][i].re = cos(M_PI * Ka * ta[i] * ta[i]);
      H[0][i].im = sin(M_PI * Ka * ta[i] * ta[i]);
    } else {
      H[0][i].re = 0;
      H[0][i].im = 0;
    }
  }

  size_t fast = 512;
  size_t slow = 256;
  bool forwardTrans = true;
  zip_op_t op = ZIP_MULT;

  DASH_FFT_flt(g[0], g2[0], fast, forwardTrans);

  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  uint32_t completion_ctr = 0;
  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (i = 0; i < Nslow; i++) {
    DASH_FFT_flt_nb(&s0[i], &fft_out_0[i], &fast, &forwardTrans, &barrier);
  }

  while (completion_ctr != Nslow) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  for (i = 0; i < Nslow; i++) {
    fftshift(fft_out_0[i], Nfast);
  }

  cond = PTHREAD_COND_INITIALIZER;
  mutex = PTHREAD_MUTEX_INITIALIZER;
  completion_ctr = 0;
  barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (i = 0; i < Nslow; i++) {
    DASH_ZIP_flt_nb(&fft_out_0[i], &g2[0], &fft_inp_1[i], &fast, &op, &barrier);
/*
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_1[j].re = (fft_out_0+i*Nfast)[j].re * g2[j].re - (fft_out_0+i*Nfast)[j].im * g2[j].im;
      fft_inp_1[j].im = (fft_out_0+i*Nfast)[j].im * g2[j].re + (fft_out_0+i*Nfast)[j].re * g2[j].im;
    }
*/
  }

  while (completion_ctr != Nslow) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  forwardTrans = false;

  cond = PTHREAD_COND_INITIALIZER;
  mutex = PTHREAD_MUTEX_INITIALIZER;
  completion_ctr = 0;
  barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (i = 0; i < Nslow; i++) {
    DASH_FFT_flt_nb(&fft_inp_1[i], &fft_out_1[i], &fast, &forwardTrans, &barrier);
  }

  while (completion_ctr != Nslow) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  for (i = 0; i < Nslow; i++) {
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_3[j][i].re = fft_out_1[i][j].re;
      fft_inp_3[j][i].im = fft_out_1[i][j].im;
    }
  }

  // Azimuth FFT
  forwardTrans = true;

  cond = PTHREAD_COND_INITIALIZER;
  mutex = PTHREAD_MUTEX_INITIALIZER;
  completion_ctr = 0;
  barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (i = 0; i < Nfast; i++) {
    DASH_FFT_flt_nb(&fft_inp_3[i], &S1[i], &slow, &forwardTrans, &barrier);
  }

  while (completion_ctr != Nfast) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  for (i = 0; i < Nfast; i++) {
    fftshift(S1[i], Nslow);
  }

  cond = PTHREAD_COND_INITIALIZER;
  mutex = PTHREAD_MUTEX_INITIALIZER;
  completion_ctr = 0;
  barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  // Azimuth Compression
  for (i = 0; i < Nfast; i++) {
    DASH_ZIP_flt_nb(&S1[i], &H[0], &fft_inp_2[i], &slow, &op, &barrier);
/*
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_2[j].re = (S1+i*Nslow)[j].re * H[j].re - (S1+i*Nslow)[j].im * H[j].im;
      fft_inp_2[j].im = (S1+i*Nslow)[j].im * H[j].re + (S1+i*Nslow)[j].re * H[j].im;
    }
*/
  }

  while (completion_ctr != Nfast) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  forwardTrans = false;

  cond = PTHREAD_COND_INITIALIZER;
  mutex = PTHREAD_MUTEX_INITIALIZER;
  completion_ctr = 0;
  barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
  pthread_mutex_lock(barrier.mutex);

  for (i = 0; i < Nfast; i++) {
    DASH_FFT_flt_nb(&fft_inp_2[i], &fft_out_2[i], &slow, &forwardTrans, &barrier);
  }

  while (completion_ctr != Nfast) {
    pthread_cond_wait(barrier.cond, barrier.mutex);
  }
  pthread_mutex_unlock(barrier.mutex);

  for (i = 0; i < Nfast; i++) {
    fftshift(fft_out_2[i], Nslow);
    for (j = 0; j < Nslow; j++) {
      sac[i+j*Nfast] = sqrt(fft_out_2[i][j].re * fft_out_2[i][j].re + fft_out_2[i][j].im * fft_out_2[i][j].im);
    }
  }
  
  fp = fopen("./output/SAR_output.txt", "w");
  if (fp != NULL) {
    for (i = 0; i < Nslow; i++) {
      for (j = 0; j < Nfast; j++) {
        fprintf(fp, "%lf ", sac[j + i * Nfast]);
      }
      fprintf(fp, "\n");
      fflush(fp);
    }
    fclose(fp);
  }
  //printf("[SAR] Execution is complete...\n");
  
  return 0;
}
