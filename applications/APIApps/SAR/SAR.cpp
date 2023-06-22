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
  dash_cmplx_flt_type *fft_out_0;
  dash_cmplx_flt_type *fft_inp_1;
  dash_cmplx_flt_type *fft_out_1;
  dash_cmplx_flt_type *fft_inp_2;
  dash_cmplx_flt_type *fft_out_2;
  dash_cmplx_flt_type *fft_inp_3;

  FILE *fp;

  double *ta;

  double Rmin, Rmax;
  double *tr;

  dash_cmplx_flt_type *g;
  dash_cmplx_flt_type *g2;
  dash_cmplx_flt_type *H;

  dash_cmplx_flt_type *s0;
  dash_cmplx_flt_type *S1;

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

  R0 = sqrt(Yc * Yc + h * h);
  Ka = 2 * v * v / lambda / R0;
  s0 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_out_0 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_inp_1 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_out_1 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_inp_2 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_out_2 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  fft_inp_3 = (dash_cmplx_flt_type*) malloc(Nslow * Nfast * sizeof(dash_cmplx_flt_type));
  sac = (double*) malloc(Nslow * Nfast * sizeof(double));

  fp = fopen(RAWDATA, "r");
  for (i = 0; i < Nslow * Nfast; i++) {
    fscanf(fp, "%f", &s0[i].re);
    fscanf(fp, "%f", &s0[i].im);
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
  g = (dash_cmplx_flt_type*) malloc(Nfast * sizeof(dash_cmplx_flt_type));
  g2 = (dash_cmplx_flt_type*) malloc(Nfast * sizeof(dash_cmplx_flt_type));
  for (i = 0; i < Nfast; i += 1) {
    if (tr[i] > -Tr / 2 && tr[i] < Tr / 2) {
      g[i].re = cos(M_PI * Kr * tr[i] * tr[i]);
      g[i].im = -sin(M_PI * Kr * tr[i] * tr[i]);
    } else {
      g[i].re = 0;
      g[i].im = 0;
    }
  }

  H = (dash_cmplx_flt_type*) malloc(Nslow * sizeof(dash_cmplx_flt_type));
  for (i = 0; i < Nslow; i += 1) {
    if (ta[i] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
        ta[i] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
      H[i].re = cos(M_PI * Ka * ta[i] * ta[i]);
      H[i].im = sin(M_PI * Ka * ta[i] * ta[i]);
    } else {
      H[i].re = 0;
      H[i].im = 0;
    }
  }

  size_t fast = 512;
  size_t slow = 256;
  bool forwardTrans = true;

  KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
  DASH_FFT_flt(g, g2, fast, forwardTrans);
  KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
  // printf("[SAR] Kernel execution is complete!\n");

  for (i = 0; i < Nslow; i++) {
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
    DASH_FFT_flt((s0+i*Nfast), (fft_out_0+i*Nfast), fast, true);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));

    fftshift((fft_out_0+i*Nfast), Nfast);

    KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
    DASH_ZIP_flt(&(fft_out_0[i*Nfast]), g2, &(fft_inp_1[i*Nfast]), Nfast, ZIP_MULT);
/*
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_1[j].re = (fft_out_0+i*Nfast)[j].re * g2[j].re - (fft_out_0+i*Nfast)[j].im * g2[j].im;
      fft_inp_1[j].im = (fft_out_0+i*Nfast)[j].im * g2[j].re + (fft_out_0+i*Nfast)[j].re * g2[j].im;
    }
*/
    KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));

    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
    DASH_FFT_flt(&(fft_inp_1[i*Nfast]), &(fft_out_1[i*Nfast]), fast, false);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
  }
  for (i = 0; i < Nslow; i++) {
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_3[j * Nslow + i].re = fft_out_1[i*Nfast + j].re;
      fft_inp_3[j * Nslow + i].im = fft_out_1[i*Nfast + j].im;
    }
  }

  // Azimuth FFT
  S1 = (dash_cmplx_flt_type*) malloc(Nfast * Nslow * sizeof(dash_cmplx_flt_type));
  for (i = 0; i < Nfast; i++) {
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
    DASH_FFT_flt(&(fft_inp_3[i*Nslow]), &(S1[i*Nslow]), slow, true);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));

    fftshift((S1+i*Nslow), Nslow);
  }

  // Azimuth Compression
  for (i = 0; i < Nfast; i++) {
    //KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
    DASH_ZIP_flt((S1+i*Nslow), H, (fft_inp_2+i*Nslow), Nslow, ZIP_MULT);
/*
    for (j = 0; j < Nfast; j += 1) {
      fft_inp_2[j].re = (S1+i*Nslow)[j].re * H[j].re - (S1+i*Nslow)[j].im * H[j].im;
      fft_inp_2[j].im = (S1+i*Nslow)[j].im * H[j].re + (S1+i*Nslow)[j].re * H[j].im;
    }
*/
    //KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));

    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
    DASH_FFT_flt((fft_inp_2+i*Nslow), (fft_out_2+i*Nslow), slow, false);
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));

    fftshift((fft_out_2+i*Nslow), Nslow);
    for (j = 0; j < Nslow; j++) {
      sac[i+j*Nfast] = sqrt(fft_out_2[i*Nslow+j].re * fft_out_2[i*Nslow+j].re + fft_out_2[i*Nslow+j].im * fft_out_2[i*Nslow+j].im);
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
  printf("[SAR] Execution is complete...\n");
  
  return 0;
}
