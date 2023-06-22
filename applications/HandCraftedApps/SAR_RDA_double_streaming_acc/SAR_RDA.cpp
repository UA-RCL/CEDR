#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include "SAR_RDA.hpp"
#include "../include/DashExtras.h"
#include "../include/gsl_fft_mex.c"
#include "../include/gsl_ifft_mex.c"
#include <cstdint>
#include <dlfcn.h>

//#define PROGPATH DASH_DATA "Dash-RadioCorpus/SAR_RDA/"
//#define PROGPATH DASH_DATA
#define PROGPATH "./input/"
#define RAWDATA PROGPATH "rawdata_rda.txt"
#define OUTPUT "SAR_RDA-output.txt"

/***************************/
double *ta;
double *trng;

double *g;
double *g2;

double *S1;

double *src;
double *H;

double *sac;
double *s0;
/***************************/

double *_ta;
double *_trng;

double *_g;
double *_g2;

double *_S1;

double *_src;
double *_H;


double R0;
double Ka;

double *_sac;
double *_s0;

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
double c;
double Rmin, Rmax;

// Pointer to use to hold the shared object file handle
void *dlhandle;
void (*dash_fft_func)(double**, double**, size_t*, bool*);

void __attribute__((constructor)) setup(void) {	
  printf("[SAR_RDA] intializing variables\n");
  
  c = 3e8;

	
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
 
/*******************************************************/ 
  ta = (double *) malloc(Nslow * sizeof(double));
  trng = (double *) malloc(Nfast * sizeof(double));
	
  g = (double *) malloc(2 * Nfast * sizeof(double));
  g2 = (double *) malloc(2 * Nfast * sizeof(double));
  
  S1 = (double *) malloc(2 * Nfast * Nslow * sizeof(double)); 
  
  H = (double *) malloc(2 * Nslow * sizeof(double));
  
  src = (double *) malloc(2 * Nfast * Nslow * sizeof(double)); 
  
  s0 = (double *) malloc(2 * Nslow * Nfast * sizeof(double)); //SAR_LFM_2 & SAR_node_1
  sac = (double *) malloc(Nslow * Nfast * sizeof(double));  
/*******************************************************/

  _ta = (double *) malloc(Nslow * sizeof(double));
  _trng = (double *) malloc(Nfast * sizeof(double));
	
  _g = (double *) malloc(2 * Nfast * sizeof(double));
  _g2 = (double *) malloc(2 * Nfast * sizeof(double));
  
  _S1 = (double *) malloc(2 * Nfast * Nslow * sizeof(double)); 
  
  _H = (double *) malloc(2 * Nslow * sizeof(double));
  
  _src = (double *) malloc(2 * Nfast * Nslow * sizeof(double));  
  
  _s0 = (double *) malloc(2 * Nslow * Nfast * sizeof(double)); //SAR_LFM_2 & SAR_node_1
  _sac = (double *) malloc(Nslow * Nfast * sizeof(double));
/*******************************************************/ 
  
  Rmin = sqrt((Yc - Y0) * (Yc - Y0) + h * h);
  Rmax = sqrt((Yc + Y0) * (Yc + Y0) + h * h);
  
  
  ///////////////accelerator////////////////////////////////
	dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
	if (dlhandle == nullptr) {
		printf("Unable to open FFT shared object!\n");
	} else {
		dash_fft_func = (void(*)(double**, double**, size_t*, bool*)) dlsym(dlhandle, "DASH_FFT_fft");
		if (dash_fft_func == nullptr) {
			printf("Unable to get function handle for DASH_FFT_fft accelerator function!\n");
		}
	}
  
  printf("[SAR_RDA] intialization done\n");
}

void __attribute__((destructor)) clean_app(void){
	printf("[SAR_RDA] destroying variables\n");
/*******************************************************/  	
	free(ta);
	free(trng);
	
	free(g);
	free(g2);
	
	free(S1);
	
	free(H);
	
	free(src);
	
	free(sac);
	free(s0);
/*******************************************************/  

	free(_ta);
	free(_trng);
	
	free(_g);
	free(_g2);
	
	free(_S1);
	
	free(_H);
	
	free(_src);
	
	free(_sac);
	free(_s0);

	if (dlhandle != nullptr) {
		dlclose(dlhandle);
	}

/*******************************************************/  
	
	printf("[SAR_RDA] destruction done\n");
}

/* Function Declarations */
//void swap(double *, double *);
//void fftshift(double *, double);

/**************** Kernels ****************/
void swap(double *v1, double *v2) {
  float tmp = *v1;
  *v1 = *v2;
  *v2 = tmp;
}

void fftshift(double *data, double count) {
  int k;
  int c = (double)floor((float)count / 2);
  // For odd and for even numbers of element use different algorithm
  if ((int)count % 2 == 0) {
    for (k = 0; k < 2 * c; k += 2) {
      swap(&data[k], &data[k + 2 * c]);
      swap(&data[k + 1], &data[k + 1 + 2 * c]);
    }
  } else {
    double tmp1 = data[0];
    double tmp2 = data[1];
    for (k = 0; k < 2 * c; k += 2) {
      data[k] = data[2 * c + k + 2];
      data[k + 1] = data[2 * c + k + 3];
      data[2 * c + k + 2] = data[k + 2];
      data[2 * c + k + 3] = data[k + 3];
    }
    data[2 * c] = tmp1;
    data[2 * c] = tmp2;
  }
}

/**************** Tasks **********************/
extern "C" void SAR_node_head(void){
}

extern "C" void SAR_LFM_1(void){
  trng[0] = 0; 
  static bool select1 = true;
  
  if(select1){
	  // Create range vector
	  for (int i = 1; i < Nfast; i++) {
		trng[i] = trng[i - 1] + (2 * Rmax / c + Tr - 2 * Rmin / c) / (Nfast - 1);
	  } 
	  
	  for (int i = 0; i < 2 * Nfast; i += 2) {
		if (trng[i / 2] > -Tr / 2 && trng[i / 2] < Tr / 2) {
		  g[i] = cos(M_PI * Kr * trng[i / 2] * trng[i / 2]);
		  g[i + 1] = -sin(M_PI * Kr * trng[i / 2] * trng[i / 2]);
		} else {
		  g[i] = 0;
		  g[i + 1] = 0;
		}
		
	  }
	  //  KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		  gsl_fft(g, g2, Nfast);
	  //  KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));	  
  } else {
	    // Create range vector
	  for (int i = 1; i < Nfast; i++) {
		_trng[i] = _trng[i - 1] + (2 * Rmax / c + Tr - 2 * Rmin / c) / (Nfast - 1);
	  } 
	  
	  for (int i = 0; i < 2 * Nfast; i += 2) {
		if (_trng[i / 2] > -Tr / 2 && _trng[i / 2] < Tr / 2) {
		  _g[i] = cos(M_PI * Kr * _trng[i / 2] * _trng[i / 2]);
		  _g[i + 1] = -sin(M_PI * Kr * _trng[i / 2] * _trng[i / 2]);
		} else {
		  _g[i] = 0;
		  _g[i + 1] = 0;
		}
	  }
	  
	  //  KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		  gsl_fft(_g, _g2, Nfast);
	  //  KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
  }
  select1 = !select1;
 
}

extern "C" void SAR_LFM_2(void){
  double R0;
  double Ka;
  
  int i;
  FILE *fp;
  FILE *_fp;
  
  static bool select2 = true;
  
  R0 = sqrt(Yc * Yc + h * h);
  Ka = 2 * v * v / lambda / R0;
  ta[0] = 0;
  
  /* Read in raw radar data */
  if(select2){
	  fp = fopen(RAWDATA, "r");
	  for (i = 0; i < 2 * Nslow * Nfast; i++) {
		fscanf(fp, "%lf", &s0[i]);
	  }	 
	   fclose(fp);	
	  // Create azimuth vector
	  for (i = 1; i < Nslow; i++) {
		ta[i] = ta[i - 1] + (Xmax - Xmin) / v / (Nslow - 1);
	  }
	  
	   // Azimuth Compression
	  for (i = 0; i < 2 * Nslow; i += 2) {
		if (ta[i / 2] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
			ta[i / 2] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
		  H[i] = cos(M_PI * Ka * ta[i / 2] * ta[i / 2]);
		  H[i + 1] = sin(M_PI * Ka * ta[i / 2] * ta[i / 2]);
		} else {
		  H[i] = 0;
		  H[i + 1] = 0;
		}
	  }
	   
  } else {
	  _fp = fopen(RAWDATA, "r");
	  for (i = 0; i < 2 * Nslow * Nfast; i++) {
		fscanf(_fp, "%lf", &_s0[i]);
	  }  
	    fclose(_fp);
		  // Create azimuth vector
	  for (i = 1; i < Nslow; i++) {
		_ta[i] = _ta[i - 1] + (Xmax - Xmin) / v / (Nslow - 1);
	  }
	  
	   // Azimuth Compression
	  for (i = 0; i < 2 * Nslow; i += 2) {
		if (_ta[i / 2] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
			_ta[i / 2] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
		  _H[i] = cos(M_PI * Ka * _ta[i / 2] * _ta[i / 2]);
		  _H[i + 1] = sin(M_PI * Ka * _ta[i / 2] * _ta[i / 2]);
		} else {
		  _H[i] = 0;
		  _H[i + 1] = 0;
		}
	  }
  }

  select2 = !select2;
} 

extern "C" void SAR_node_1_cpu(void){
  printf("[SAR_RDA] SAR_node_1 execution begun\n");
  double *fft_arr;
  double *temp;
  double *temp2;
  double *temp3;	
  
  int i, j;
  
  static bool select3 = true;
	
  fft_arr = (double*) malloc(2 * Nfast * sizeof(double));
  temp = (double*) malloc(2 * Nfast * sizeof(double));
  temp2 = (double*) malloc(2 * Nfast * sizeof(double));
  temp3 = (double*) malloc(2 * Nfast * sizeof(double));	
	
  if(select3){
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		gsl_fft(fft_arr, temp, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * g2[j] - temp[j + 1] * g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * g2[j] + temp[j] * g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		gsl_ifft(temp2, temp3, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  src[j * Nslow + 2 * i] = temp3[j];
		  src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }
  } else {
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = _s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		gsl_fft(fft_arr, temp, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * _g2[j] - temp[j + 1] * _g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * _g2[j] + temp[j] * _g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		gsl_ifft(temp2, temp3, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  _src[j * Nslow + 2 * i] = temp3[j];
		  _src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }	  
  }
 
  
  free(fft_arr);
  free(temp);
  free(temp2);
  free(temp3);
  select3 = !select3;  
  printf("[SAR_RDA] SAR_node_1 done\n");

}

extern "C" void SAR_node_1_acc(void){
  printf("[SAR_RDA] SAR_node_1 execution begun\n");
  double *fft_arr;
  double *temp;
  double *temp2;
  double *temp3;	
  
  int i, j;
  
  static bool select3 = true;
  size_t size = Nfast;
  bool fwdFFT = true;
  bool invFFT = false;
	
  fft_arr = (double*) malloc(2 * Nfast * sizeof(double));
  temp = (double*) malloc(2 * Nfast * sizeof(double));
  temp2 = (double*) malloc(2 * Nfast * sizeof(double));
  temp3 = (double*) malloc(2 * Nfast * sizeof(double));	
	
  if(select3){
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		//gsl_fft(fft_arr, temp, Nfast);
		(*dash_fft_func)(&fft_arr, &temp, &size, &fwdFFT);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * g2[j] - temp[j + 1] * g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * g2[j] + temp[j] * g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		//gsl_ifft(temp2, temp3, Nfast);
		(*dash_fft_func)(&temp2, &temp3, &size, &invFFT);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  src[j * Nslow + 2 * i] = temp3[j];
		  src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }
  } else {
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = _s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		(*dash_fft_func)(&fft_arr, &temp, &size, &fwdFFT);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * _g2[j] - temp[j + 1] * _g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * _g2[j] + temp[j] * _g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		//gsl_ifft(temp2, temp3, Nfast);
		(*dash_fft_func)(&temp2, &temp3, &size, &invFFT);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  _src[j * Nslow + 2 * i] = temp3[j];
		  _src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }	  
  }
 
  
  free(fft_arr);
  free(temp);
  free(temp2);
  free(temp3);
  select3 = !select3;  
  printf("[SAR_RDA] SAR_node_1 done\n");

}

extern "C" void SAR_node_2_cpu(void){
  double *temp4;
  double *fft_arr_2;
  int i, j;
  
  static bool select4 = true;
  
  fft_arr_2 = (double*) malloc(2 * Nslow * sizeof(double));
  temp4 = (double*) malloc(2 * Nslow * sizeof(double));
  
  if(select4){
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		gsl_fft(fft_arr_2, temp4, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  S1[j + i * 2 * Nslow] = temp4[j];
		  S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }	  
  } else {
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = _src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = _src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		gsl_fft(fft_arr_2, temp4, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  _S1[j + i * 2 * Nslow] = temp4[j];
		  _S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  }
  
  
  free(fft_arr_2);
  free(temp4);
  select4 = !select4;
}

extern "C" void SAR_node_2_acc(void){
  double *temp4;
  double *fft_arr_2;
  int i, j;
  size_t len = 256;
  bool isFwd = true;
  
  static bool select4 = true;
  
  fft_arr_2 = (double*) malloc(2 * Nslow * sizeof(double));
  temp4 = (double*) malloc(2 * Nslow * sizeof(double));
  
  if(select4){
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		(*dash_fft_func)(&fft_arr_2, &temp4, &len, &isFwd);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  S1[j + i * 2 * Nslow] = temp4[j];
		  S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  } else {
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = _src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = _src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		(*dash_fft_func)(&fft_arr_2, &temp4, &len, &isFwd);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  _S1[j + i * 2 * Nslow] = temp4[j];
		  _S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  }
  
  free(fft_arr_2);
  free(temp4);
  select4 = !select4;
}

extern "C" void SAR_node_3_cpu(void){
  double *fft_arr_4;
  double *temp8;
  double *temp9;	
  
  int i, j;
  
  static bool select5 = true;
  
  fft_arr_4 = (double*) malloc(2 * Nslow * sizeof(double));
  temp8 = (double*) malloc(2 * Nslow * sizeof(double));
  temp9 = (double*) malloc(2 * Nslow * sizeof(double));

  if(select5){
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		gsl_ifft(fft_arr_4, temp9, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }  
  } else {
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = _S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		gsl_ifft(fft_arr_4, temp9, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  _sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }		
  }	
  free(fft_arr_4);
  free(temp8);
  free(temp9);
  select5 = !select5;
}

extern "C" void SAR_node_3_acc(void){
  double *fft_arr_4;
  double *temp8;
  double *temp9;	
  
  int i, j;

  size_t len = 256;
  bool isFwd = false;
  
  static bool select5 = true;
  
  fft_arr_4 = (double*) malloc(2 * Nslow * sizeof(double));
  temp8 = (double*) malloc(2 * Nslow * sizeof(double));
  temp9 = (double*) malloc(2 * Nslow * sizeof(double));

  if(select5){
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		(*dash_fft_func)(&fft_arr_4, &temp9, &len, &isFwd);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }  
  } else {
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = _S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		(*dash_fft_func)(&fft_arr_4, &temp9, &len, &isFwd);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  _sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }		
  }	
  free(fft_arr_4);
  free(temp8);
  free(temp9);
  select5 = !select5;
}

extern "C" void SAR_node_4(void){
  /* Write out image */
  FILE *fp1;
  FILE *_fp1;
  
  static bool select6 = true;
  
  if(select6){
	  fp1 = fopen(OUTPUT, "w");
	  for (int i = 0; i < Nslow; i++) {
		for (int j = 0; j < Nfast; j++) {
		  fprintf(fp1, "%lf ", sac[j + i * Nfast]);
		}
		fprintf(fp1, "\n");
		fflush(fp1);
	  }
	  fclose(fp1);	  
  } else {
	  _fp1 = fopen(OUTPUT, "w");
	  for (int i = 0; i < Nslow; i++) {
		for (int j = 0; j < Nfast; j++) {
		  fprintf(_fp1, "%lf ", _sac[j + i * Nfast]);
		}
		fprintf(_fp1, "\n");
		fflush(_fp1);
	  }
	  fclose(_fp1);	  
  }
  select6 = !select6;	
}


int main(int argc, char *argv[]) {
}