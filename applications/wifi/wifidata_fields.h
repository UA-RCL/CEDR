#include "viterbi.h"
#include "baseband_lib.h"

struct wifitx_fields {
	
     int           user_data_len;
     int           encoder_id;
     int           output_len;
     int           fft_id;
     int           fft_n;
     int           hw_fft_busy;
     int           cyclic_length;
     int           sym_num;

     unsigned char inbit[1024];
     unsigned char scram[1024];
     signed   char enc_out[OUTPUT_LEN];
     unsigned char intl_out[OUTPUT_LEN];
     double        sig_real[OUTPUT_LEN];
     double        sig_img[OUTPUT_LEN];
     float         in_ifft[FFT_N*2];
     comp_t        pilot_out[FFT_N];
     comp_t		   pilot_infft[FFT_N];
     comp_t        cyclic_out[SYM_NUM*TOTAL_LEN];
};

struct wifirx_fields {
    
    char   rate;
    int    dId;
    int    descrambler_n;
    int    pilot_len;
    int    fft_id;
    int    decoderId;
    int    fft_n;
    int    demod_n;
    int    deinterleaver_n;
    int    sym_num;

    signed    char                   dec_in[OUTPUT_LEN];
    unsigned  char                   dec_out[USR_DAT_LEN];
    signed    char                   dec_pun_out[OUTPUT_LEN];
    signed    char                   deintl_out[OUTPUT_LEN];
    double    out_real[OUTPUT_LEN];
    double    out_img[OUTPUT_LEN];
    signed    char                   outbit[OUTPUT_LEN];
    unsigned  char                   descram[USR_DAT_LEN*SYM_NUM+1];
    int       hw_fft_busy;
    
    comp_t    out_fd[FFT_N];
	comp_t    pilot_rm[INPUT_LEN];
    float     *offt;
    float     pilotdata_rx[PILOT_LEN];
    float     out_fft_c2f[INPUT_LEN * 2];

    comp_t    rfIoBuf[BUF_LIMIT];
    viterbiDecoder_t vDecoder[MAX_VITERBI];

    float mBuf[6];
    int maxIndex;
    long int prtMaxIdx;
    int payloadStart, payloadPrtStart;
    int spRate;
    int sIdx;
    long int sampleCount;
};

