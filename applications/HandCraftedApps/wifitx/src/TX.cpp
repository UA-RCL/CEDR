// Created by sahilhassan on 3/30/21.
//
// OUTLINE:
// ============================================
// Constructor
  // Initialization of global memory and decoder/encoders

// Head node
  // create_rfInt [just configures stuff]
  // read_config [reads from input/tx.cfg file]
  // generate_input
// FOR LOOP optimized in parallel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // scrambler [Takes one frame (8B) and scrambles and generates (8B) output]
  // viterbi_encoder (output size is (8B + 6B zero padding)*2, probably some kind of complex number)
  // viterbi_puncturing (takes encoder output as input and produces same dimension output)
  // interleave (takes punctured output as input and produces same dimension output))
  // MOD_QPSK
  // pilotInsertion
  // ifft_hs
  // cyclicPrefix
// FOR LOOP ends here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Postprocessing-
  // zero padding
  // preamble
  // frame duplication
// Send data/ output write file
// ==========================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include "common.h"
#include "IFFT_FFT.h"
#include "scrambler_descrambler.h"
#include "CyclicPrefix.h"
#include "Preamble_ST_LG.h"
#include "viterbi.h"
#include "baseband_lib.h"
#include "interleaver_deintleaver.h"
#include "qpsk_Mod_Demod.h"
#include "datatypeconv.h"
#include "pilot.h"
#include "fft_hs.h"
#include "txData.h"
#include "rf_interface.h"
#include "rfInf.h"
#include <errno.h>
#include <limits.h>
#include <cstdint>
#include <dlfcn.h>

#include <fftw3.h>
#include <complex>


//typedef double complex cplx;
typedef std::complex<double> cplx;
const std::complex<double> I (0.0, 1.0);
//#include "DashExtras.h"
//#include "gsl_fft_mex.c"
//#include "gsl_ifft_mex.c"


// Declare global pointers
int txOption;
int fft_id, hw_fft_busy;
char rate;
int encoderId;
int frameLength;
unsigned char* inbit;
unsigned char* scram;                 // MULTIPLE FRAMES- SINGLE POINTER
signed char* enc_out;                 // MULTIPLE FRAMES- SINGLE POINTER
signed char* enc_dep_out;             // MULTIPLE FRAMES- SINGLE POINTER
unsigned char* intl_out;              // MULTIPLE FRAMES- SINGLE POINTER
double * sig_real; double* sig_img;   // MULTIPLE FRAMES- SINGLE POINTER
float *in_ifft;                       // MULTIPLE FRAMES- SINGLE POINTER
comp_t **ifft_in;                     // MULTIPLE FRAMES- DOUBLE POINTER
comp_t *pilot_out;                    // MULTIPLE FRAMES- SINGLE POINTER
double * ifft_out;
comp_t *cyclic_out;                   // MULTIPLE FRAMES- SINGLE POINTER
comp_t *pre_out;
comp_t *txData;

// FFTW3
fftwf_complex *ifft3_in;
fftwf_complex *ifft3_out;
fftwf_plan p;

int user_data_len;

void *dlhandle;
void (*ifft_accel_func)(double**, double**, size_t*, bool*, uint8_t);

// Declare visible TASK_ID
__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_TASK_ID__ = 0;
__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_CLUSTER_IDX__ = 0;
// Constructor
void __attribute__((constructor)) setup(void){
  printf("[WiFi-TX] Running Constructor!\n");
  // FFTW3 ------
  ifft3_in = (fftwf_complex*)malloc( FFT_N * sizeof(fftwf_complex) );
  ifft3_out = (fftwf_complex*)malloc( FFT_N * sizeof(fftwf_complex) );
  p = fftwf_plan_dft_1d(FFT_N, ifft3_in, ifft3_out, FFTW_BACKWARD, FFTW_ESTIMATE);

  rate = PUNC_RATE_1_2;
  fft_id = 1; hw_fft_busy = 1;
  printf("[WiFi-TX] Initializing Encoder!\n");
  init_viterbiEncoder();
  encoderId = get_viterbiEncoder();
  set_viterbiEncoder(encoderId);

  frameLength = 256 + PREAMBLE_LEN + SYM_NUM*(TOTAL_LEN);  // Revise 256

  inbit = (unsigned char*) calloc(1024, sizeof(unsigned char)); // Revise for generalized size based on SYM_NUM
  scram = (unsigned char*) calloc(SYM_NUM*1024, sizeof(unsigned char *)); // Revise for generalized size based on SYM_NUM
  enc_out = (signed char*)calloc(SYM_NUM*OUTPUT_LEN, sizeof(signed char *));
  enc_dep_out = (signed char*)calloc(SYM_NUM*OUTPUT_LEN, sizeof (signed char*));
  intl_out = (unsigned char*)calloc(SYM_NUM*OUTPUT_LEN, sizeof(unsigned char));
  sig_real = (double *)calloc(SYM_NUM*OUTPUT_LEN, sizeof(double));
  sig_img = (double *)calloc(SYM_NUM*OUTPUT_LEN, sizeof(double));

  in_ifft =(float*)calloc(SYM_NUM*FFT_N*2, sizeof(float));
  ifft_in =(comp_t**)calloc(SYM_NUM, sizeof(comp_t *));

  pilot_out = (comp_t*)calloc(SYM_NUM*FFT_N, sizeof(comp_t));
  ifft_out = (double *)calloc(2*FFT_N*SYM_NUM, sizeof(double));
  cyclic_out = (comp_t*)calloc(SYM_NUM*TOTAL_LEN, sizeof(comp_t));

  pre_out = (comp_t*) calloc(SYM_NUM*TOTAL_LEN + PREAMBLE_LEN + 2048, sizeof(comp_t));
  txData = (comp_t*) calloc(TX_DATA_LEN, sizeof(comp_t));
  
  dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open libdash-rt shared object!\n");
  } else {
    ifft_accel_func = (void(*)(double**, double**, size_t*, bool*, uint8_t)) dlsym(dlhandle, "DASH_FFT_fft");
    if (ifft_accel_func == nullptr) {
      printf("Unable to get function handle for DASH_FFT_fft accelerator function!\n");
    }
    else {
      printf("Successfully opened DASH_FFT_fft accelerator function!\n");
    }
  }
  

  printf("[WiFi-TX] Finished Constructor!\n");
}

// Destructor
void __attribute__((destructor)) teardown(void){
  printf("[WiFi-TX] Running Destructor!\n");
  // FFTW3-------
  fftwf_free(ifft3_in);
  fftwf_free(ifft3_out);
  fftwf_destroy_plan(p);

  free(inbit);
  free(scram);
  free(enc_out);
  free(enc_dep_out);
  free(intl_out);
  free(sig_real); free(sig_img);
  free(in_ifft);

  free(ifft_in);
  free(pilot_out);
  free(cyclic_out);
  free(pre_out);
  free(txData);
  
  dlclose(dlhandle);
  printf("[WiFi-TX] Finished Destructor!\n");
}

// ==================================== User-defined functions ==========================================
void readConfig() {
  FILE *cfp;
  char buf[1024];

  cfp = fopen("./input/tx.cfg", "r");
  if(cfp == NULL) {  // Unclear about the 'if' statement instructions
    char currWorkingDir[PATH_MAX];    // SH: Where is PATH_MAX coming from
    printf("fail to open config file: %s\n", strerror(errno));
    getcwd(currWorkingDir, PATH_MAX); // SH: Where is getcwd definition
    printf("current working dir: %s\n", currWorkingDir);
    exit(1);
  }

  fgets(buf, 1024, cfp);
  sscanf(buf, "%d", &txOption);
  printf("- %s\n", (txOption == 0) ? "Tx fixed string" : "Tx variable string");
}

void random_wait_time(int random_wait_time_in_us) {
  for(int k = 0; k < random_wait_time_in_us; k++) {
    for(int i = 0; i < 170; i++);
  }
}

void fftwf_fft(double *input_array, fftwf_complex *in, fftwf_complex *out, double *output_array, size_t n_elements, fftwf_plan p )
{
  for(size_t i = 0; i < n_elements; i++)
  {
    in[i][0] = input_array[2*i];
    in[i][1] = input_array[(2*i)+1];
  }
  fftwf_execute(p);
  for(size_t i = 0; i < n_elements; i++)
  {
    output_array[(2*i)] = out[i][0];
    output_array[(2*i)+1] = out[i][1];
  }
}

// ========================================================================================================


extern "C" void WIFITX_HEAD(void) {
  printf("[WiFi-TX] Reading Configuration File\n");
  readConfig();
  printf("[WiFi-TX] Generating input data\n");
  // input data generation
  user_data_len = txDataGen(txOption, inbit, SYM_NUM); // Data length in bits
}

extern "C" void WIFITX_SCRAMBLER(void){
  printf("[WiFi-TX] Executing Scrambler with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - 1;
  scrambler(USR_DAT_LEN, &inbit[taskId*USR_DAT_LEN], &scram[1024*taskId]);
}

extern "C" void WIFITX_ENCODER(void){
  printf("[WiFi-TX] Executing Encoder with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*1) - 1;
  viterbi_encoding(encoderId, &scram[1024*taskId], &enc_out[taskId*OUTPUT_LEN]);
}

extern "C" void WIFITX_PUNCTURING(void){
  printf("[WiFi-TX] Executing Viterbi Puncturing with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*2) - 1;
  viterbi_puncturing(rate, &enc_out[taskId*OUTPUT_LEN], &enc_dep_out[taskId*OUTPUT_LEN]);
}

extern "C" void WIFITX_INTERLEAVER(void){
  printf("[WiFi-TX] Executing Interleaver with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*3) - 1;
  interleaver(&enc_dep_out[taskId*OUTPUT_LEN], OUTPUT_LEN, &intl_out[taskId*OUTPUT_LEN]);
}

extern "C" void WIFITX_MOD_QPSK(void){
  printf("[WiFi-TX] Executing MOD_QPSK with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*4) - 1;
  MOD_QPSK(OUTPUT_LEN, &intl_out[taskId*OUTPUT_LEN], &sig_real[taskId*OUTPUT_LEN], &sig_img[taskId*OUTPUT_LEN], &in_ifft[taskId*FFT_N*2]);
}

extern "C" void WIFITX_PILOTINSERTION(void){
  printf("[WiFi-TX] Executing Pilot Insertion with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*5) - 1;
  ifft_in[taskId] = (comp_t*)&in_ifft[taskId*FFT_N*2];
  pilotInsertion(ifft_in[taskId], &pilot_out[taskId*FFT_N]);
}

extern "C" void WIFITX_IFFT(void){
  printf("[WiFi-TX] Executing IFFT with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*6) - 1;
  comp_t* data = &pilot_out[taskId*FFT_N];
  double *data_in = (double*) calloc(2*FFT_N, sizeof(double));
  //double *data_out = (double*) calloc(2*FFT_N, sizeof(double));
  double *data_out = &ifft_out[taskId*2*FFT_N];
  for (int i = 0; i < FFT_N; i++) {
    data_in[2*i] = (double)data[i].real;
    data_in[2*i+1] = (double)data[i].imag;
  }

  fftwf_fft(data_in, ifft3_in, ifft3_out, data_out, FFT_N, p);
}

extern "C" void WIFITX_IFFT_accel(void){
  printf("[WiFi-TX] Executing IFFT accelerator with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*6) - 1;
  comp_t* data = &pilot_out[taskId*FFT_N];
  double *data_in = (double*) calloc(2*FFT_N, sizeof(double));
  double *data_out = &ifft_out[taskId*2*FFT_N];

  for (int i = 0; i < FFT_N; i++) {
    data_in[2*i] = (double)data[i].real; 
    data_in[2*i+1] = (double)data[i].imag;
  }
  size_t len = 128;
  bool isFwd = false;

  (*ifft_accel_func)(&data_in, &data_out, &len, &isFwd, __CEDR_CLUSTER_IDX__);
}

extern "C" void WIFITX_IFFT_SHIFT(void) {
  printf("[WiFi-TX] Executing IFFT_SHIFT with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*7) - 1;
  // Take data_out as input and write normalized output at pilot_out
  double *data_in = &ifft_out[taskId*2*FFT_N];

  cplx buf[FFT_N];
  cplx tmp;
  for (int i=0; i < FFT_N; i++) buf[i] = data_in[2*i] + data_in[2*i+1] * I;

  int n = FFT_N;
  int n2 = FFT_N/2;
  buf[0] = buf[0]/(double)n;
  buf[n2] = buf[n2]/(double)n;
  for(int i=1; i<n2; i++) {
    tmp = buf[i]/(double)n;
    buf[i] = buf[n-i]/(double)n;
    buf[n-i] = tmp;
  }

  comp_t* data_out = &pilot_out[taskId*FFT_N];
  for (int i = 0; i < FFT_N; i++) {
    data_out[i].real = (float)buf[i].real();
    data_out[i].imag = (float)buf[i].imag();
  }
}


extern "C" void WIFITX_CRC(void){
  printf("[WiFi-TX] Computing CRC with task_id = %d\n", __CEDR_TASK_ID__);
  thread_local int taskId;
  taskId = __CEDR_TASK_ID__ - (SYM_NUM*8) - 1;
  cyclicPrefix(&pilot_out[taskId*FFT_N], &cyclic_out[taskId*TOTAL_LEN], FFT_N, CYC_LEN);
  printf("[WiFi-TX] Finished symbol %d!\n", taskId+1);
}

extern "C" void WIFITX_POSTPROCESS(void){
  printf("[WiFi-TX] Executing Zero Padding!\n");
  for(int i=0; i<256; i++) { // 512 zero pad
    pre_out[i].real = pre_out[i].imag = 0;
  }
  printf("[WiFi-TX] Adding Data Payload!\n");
  preamble(cyclic_out, &pre_out[256], SYM_NUM*(TOTAL_LEN)); // 322 preamble + SYM_NUM*80

  printf("[WiFi-TX] Executing Frame duplication!\n");
  thread_local int i,j;
  for(i=0; i<TX_DATA_LEN - frameLength; i+=frameLength) {
    for(j=0; j<frameLength; j++) {
      txData[i+j].real = pre_out[j].real;
      txData[i+j].imag = pre_out[j].imag;
    }
  }

  for( ; i<TX_DATA_LEN; i++) {
    txData[i].real = 0;
    txData[i].imag = 0;
  }
}

extern "C" void WIFITX_SENDDATA(void){
  printf("[WiFi-TX] Sending Data OR Writing in file txdata_out.txt!\n");
  FILE *txdata_file;
  txdata_file = fopen("txdata_out.txt", "w");
  // Only adding support for file writing
  printf("[WiFi-TX] - RF file dump!!");
  for (int i=0; i < TX_DATA_LEN; i++){
    fprintf(txdata_file, "%f %f\n", txData[i].real, txData[i].imag);
  }
  fclose(txdata_file);
  printf("[WiFi-TX] WiFi TX chain complete!!\n\n");
}

int main(void){}

