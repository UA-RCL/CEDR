// variable namings
// arm ifdef -> pulse doppler

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>     // complex mult
#ifdef INCLUDE_MAIN  // for automatic json generation
#include <fstream>
#include <nlohmann/json.hpp>
#endif
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

/************* global variables/vectors **************/
double *ta;
double *trng;

double *g;
double *g2;

double *S1;

double *src;
double *H;

double *sac;
double *s0;
/****************************************************/

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

double *FFT_temp;
double *FFT_temp2;
double *FFT_temp3;
double *FFT_temp4;
double *FFT_temp5;
double *sac_temp;
double *sac_temp_1;

double *dump_in;
double *dump_out;

void *dlhandle;
void (*fft_accel_func256)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**);
void (*ifft_accel_func256)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**);
void (*fft_accel_func512)(double**, double**, size_t*);
void (*ifft_accel_func512)(double**, double**, size_t*);

__attribute__((__visibility__("default"))) thread_local int __CEDR_TASK_ID__ = 0;

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
 
   dump_in = (double*) malloc(Nslow * Nfast * sizeof(double));
  dump_out = (double*) malloc(Nslow * Nfast * sizeof(double));  
  
  FFT_temp = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  FFT_temp2 = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  FFT_temp3 = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  FFT_temp4 = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  FFT_temp5 = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  sac_temp = (double*) malloc(2 * Nslow * Nfast * sizeof(double));
  sac_temp_1 = (double*) malloc(Nslow * Nfast * sizeof(double));
  
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
	/*dlhandle = dlopen("./apps/fft-aarch64.so", RTLD_LAZY);
	if (dlhandle == nullptr) {
		printf("Unable to open FFT shared object!\n");
	}
	
	fft_accel_func256 = (void(*)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**))dlsym(dlhandle, "fft256_accel");
	if (fft_accel_func256 == nullptr) {
		printf("Unable to get function handle for FFT-256 accelerator function!\n");
	}  
	ifft_accel_func256 = (void(*)(int64_t*, int64_t*, int32_t*, int32_t*, double**, double**, double**))dlsym(dlhandle, "ifft256_accel");
	if (ifft_accel_func256 == nullptr) {
		printf("Unable to get function handle for IFFT-256 accelerator function!\n");
	}  	
	fft_accel_func512 = (void(*)(double**, double**, size_t*))dlsym(dlhandle, "fft512_accel_gsl_mex");
	if (fft_accel_func512 == nullptr) {
		printf("Unable to get function handle for FFT-512 accelerator function!\n");
	} 
	ifft_accel_func512 = (void(*)(double**, double**, size_t*))dlsym(dlhandle, "ifft512_accel_gsl_mex");
	if (ifft_accel_func512 == nullptr) {
		printf("Unable to get function handle for IFFT-512 accelerator function!\n");
	} */
  
  printf("[SAR_RDA] intialization done\n");
}

void __attribute__((destructor)) clean_app(void){
	printf("[SAR_RDA] destroying variables\n");
/*******************************************************/  	
	free(dump_in);
	free(dump_out);
	
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
	
	free(FFT_temp);
	free(FFT_temp2);
	free(FFT_temp3);
	free(FFT_temp4);
	free(FFT_temp5);
	free(sac_temp);
	free(sac_temp_1);
	
//	dlclose(dlhandle);
/*******************************************************/  
	
	printf("[SAR_RDA] destruction done\n");
}


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


extern "C" void SAR_node_head(void){
  printf("Finished SAR_node_head, task id %d\n", __CEDR_TASK_ID__);
  for (int k = 0; k < Nslow*Nfast; k += 1) {
    dump_in[k] = k+1;
  }  
  
}

extern "C" void SAR_RDA_LFM_1(void){
  trng[0] = 0;  
  
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
  
  gsl_fft(g, g2, Nfast);  
  printf("Finished SAR_RDA_LFM_1, task id %d\n", __CEDR_TASK_ID__);
}

extern "C" void SAR_RDA_LFM_2(void){
  double R0;
  double Ka;
  
  int i;
  FILE *fp;
  FILE *_fp;
  
  R0 = sqrt(Yc * Yc + h * h);
  Ka = 2 * v * v / lambda / R0;
  ta[0] = 0;
  
  /* Read in raw radar data */
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
  printf("Finished SAR_RDA_LFM_2, task id %d\n", __CEDR_TASK_ID__);  
} 

extern "C" void SAR_RDA_1_FFT_cpu(void)
{
    double *fft_arr;
	double *FFT_temp_loc;
	
	int index = 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp_loc = &((FFT_temp)[ x*2*Nfast]); 
	fft_arr = &((s0)[ x*2*Nfast]); 	
	
	gsl_fft(fft_arr, FFT_temp_loc, Nfast);	
	
	//printf("Finished SAR_RDA_1_FFT_cpu, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_1_FFT_accel(void)
{
    double *fft_arr;
	double *FFT_temp_loc;
	
	int index = 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp_loc = &((FFT_temp)[ x*2*Nfast]); 
	fft_arr = &((s0)[ x*2*Nfast]); 	
	
	//(*fft_accel_func512)(&fft_arr, &FFT_temp_loc, nullptr);	
	//printf("Finished SAR_RDA_1_FFT_accel, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_1_FFTSHIFT(void)
{
	double *FFT_temp_loc;
	
	int index = Nslow + 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp_loc = &((FFT_temp)[x*2*Nfast]); 
		
	fftshift(FFT_temp_loc, Nfast);
	
	//printf("Finished SAR_RDA_1_FFTSHIFT, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_1_Mul(void)
{
	double *FFT_temp_loc;
	double *FFT_temp2_loc;
	
	int index = 2*Nslow + 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp_loc = &((FFT_temp)[x*2*Nfast]); // and this part also
	FFT_temp2_loc = &((FFT_temp2)[x*2*Nfast]);
		
	for (int j = 0; j < 2 * Nfast; j += 2) {
	  FFT_temp2_loc[j] = FFT_temp_loc[j] * g2[j] - FFT_temp_loc[j + 1] * g2[j + 1];
	  FFT_temp2_loc[j + 1] = FFT_temp_loc[j + 1] * g2[j] + FFT_temp_loc[j] * g2[j + 1];
	}
	
	//printf("Finished SAR_RDA_1_Mul, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_1_IFFT_cpu(void)
{
	double *FFT_temp2_loc;
	double *FFT_temp3_loc;
	double *src_loc;
	
	int index = 3*Nslow + 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp2_loc = &((FFT_temp2)[ x*2*Nfast]); 
	FFT_temp3_loc = &((FFT_temp3)[ x*2*Nfast]);
		
	gsl_ifft(FFT_temp2_loc, FFT_temp3_loc, Nfast);

	
	//printf("Finished SAR_RDA_1_IFFT_cpu, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_1_IFFT_accel(void)
{
	double *FFT_temp2_loc;
	double *FFT_temp3_loc;
	double *src_loc;
	
	int index = 3*Nslow + 3; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp2_loc = &((FFT_temp2)[ x*2*Nfast]); 
	FFT_temp3_loc = &((FFT_temp3)[ x*2*Nfast]);
	
	//(*ifft_accel_func512)(&FFT_temp2_loc, &FFT_temp3_loc, nullptr);
	
	//printf("Finished SAR_RDA_1_IFFT_accel, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void SAR_RDA_Allign_1(void){
  int i, j;
  
  for (int i = 0; i < Nslow; i++) {  
	for (int j = 0; j < 2 * Nfast; j += 2) {
	  src[j * Nslow + 2 * i] = FFT_temp3[i*2*Nfast + j];
	  src[j * Nslow + 2 * i + 1] = FFT_temp3[i*2*Nfast + j + 1];
	}
  }  
  printf("Finished SAR_RDA_Allign_1, task id: %d\n", __CEDR_TASK_ID__);  
}

/*********************************************************************************/

extern "C" void SAR_RDA_2_FFT_cpu(void)
{
    double *fft_arr2;
	double *FFT_temp4_loc;
	
	int index = 4*Nslow + 4; 
    int x = __CEDR_TASK_ID__ - index;
	FFT_temp4_loc = &((FFT_temp4)[ x*2*Nslow]);
    fft_arr2 = &((src)[ x*2*Nslow]); 
	
	
	gsl_fft(fft_arr2, FFT_temp4_loc, Nslow);	
	
	//printf("Finished SAR_RDA_2_FFT_cpu, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_2_FFT_accel(void)
{
    double *fft_arr2;
	double *FFT_temp4_loc;
	
	int index = 4*Nslow + 4; 
    int x = __CEDR_TASK_ID__ - index;
	FFT_temp4_loc = &((FFT_temp4)[ x*2*Nslow]);
    fft_arr2 = &((src)[ x*2*Nslow]); 
	
	//(*fft_accel_func256)(nullptr, nullptr, nullptr, nullptr, nullptr, &fft_arr2, &FFT_temp4_loc);
	
	//printf("Finished SAR_RDA_2_FFT_accel, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_2_FFTSHIFT(void)
{
	double *FFT_temp4_loc;
	
	int index = Nfast + 4*Nslow + 4; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp4_loc = &((FFT_temp4)[ x*2*Nslow]);
		
	fftshift(FFT_temp4_loc, Nslow);
	
	//printf("Finished SAR_RDA_2_FFTSHIFT, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void SAR_RDA_Allign_2(void){
  printf("Finished SAR_RDA_Allign_2, task id: %d\n", __CEDR_TASK_ID__);  
}

/*********************************************************************************/

extern "C" void SAR_RDA_3_Mul(void)
{
	double *FFT_temp4_loc;
	double *H_loc;
	double *FFT_temp5_loc;
	
	int index = 2*Nfast + 4*Nslow + 5; 
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp4_loc = &((FFT_temp4)[ x*2*Nslow]); 
	FFT_temp5_loc = &((FFT_temp5)[ x*2*Nslow]);
		
	for (int j = 0; j < 2 * Nslow; j += 2) {
	  FFT_temp5_loc[j] = FFT_temp4_loc[j] * H[j] - FFT_temp4_loc[j + 1] * H[j + 1];
	  FFT_temp5_loc[j + 1] = FFT_temp4_loc[j + 1] * H[j] + FFT_temp4_loc[j] * H[j + 1];
	}
	
	//printf("Finished SAR_RDA_3_Mul, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_3_IFFT_cpu(void)
{
	double *FFT_temp5_loc;
	double *sac_temp_loc;
	
	int index = 3*Nfast + 4*Nslow + 5;
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp5_loc = &((FFT_temp5)[ x*2*Nslow]);
	sac_temp_loc = &((sac_temp)[ x*2*Nslow]);
	
	gsl_ifft(FFT_temp5_loc, sac_temp_loc, Nslow);	
	
	//printf("Finished SAR_RDA_3_IFFT_cpu, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_3_IFFT_accel(void)
{
	double *FFT_temp5_loc;
	double *sac_temp_loc;
	
	int index = 3*Nfast + 4*Nslow + 5;
    int x = __CEDR_TASK_ID__ - index;
    FFT_temp5_loc = &((FFT_temp5)[ x*2*Nslow]);
	sac_temp_loc = &((sac_temp)[ x*2*Nslow]);
	
	
	//(*ifft_accel_func256)(nullptr, nullptr, nullptr, nullptr, nullptr, &FFT_temp5_loc, &sac_temp_loc);
	
	//printf("Finished SAR_RDA_3_IFFT_accel, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_3_FFTSHIFT(void)
{
	double *sac_temp_loc;
	
	int index = 4*Nfast + 4*Nslow + 5; 
    int x = __CEDR_TASK_ID__ - index;
    sac_temp_loc = &((sac_temp)[ x*2*Nslow]);
		
	fftshift(sac_temp_loc, Nslow);
	
	//printf("Finished SAR_RDA_3_FFTSHIFT, task id: %d\n", __CEDR_TASK_ID__);
}

/*********************************************************************************/

extern "C" void SAR_RDA_3_Amplitude(void)
{
	double *sac_temp_loc;
	double *sac_temp_1_loc;
	
	int index = 5*Nfast + 4*Nslow + 5; 
    int x = __CEDR_TASK_ID__ - index;
	//printf("SAR_RDA_3_Amplitude, index: %d, x: %d\n", index, x);
    sac_temp_loc = &((sac_temp)[x*2*Nslow]);
	sac_temp_1_loc = &((sac_temp_1)[x*Nslow]);
	
	for (int j = 0; j < Nslow; j++) {
	  sac_temp_1_loc[j] = sqrt(sac_temp_loc[2 * j] * sac_temp_loc[2 * j] + sac_temp_loc[2 * j + 1] * sac_temp_loc[2 * j + 1]);
	}
	
	//printf("Finished SAR_RDA_3_Amplitude, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void SAR_RDA_FWrite(void){
 
	for (int i = 0; i < Nfast; i++) {
	  for(int j = 0; j < Nslow; j++)
	   sac[i + j * Nfast] = sac_temp_1[i*Nslow + j];	  
	}
 
  /* Write out image */
  FILE *fp1;
  
  fp1 = fopen(OUTPUT, "w");
  for (int i = 0; i < Nslow; i++) {
	for (int j = 0; j < Nfast; j++) {
	  fprintf(fp1, "%lf ", sac[j + i * Nfast]);
	}
	fprintf(fp1, "\n");
	fflush(fp1);
  }
  fclose(fp1);	 
  printf("Finished SAR_RDA_FWrite, task id: %d\n", __CEDR_TASK_ID__);  
}

/*********************************************************************************/

int main(void) {    
}