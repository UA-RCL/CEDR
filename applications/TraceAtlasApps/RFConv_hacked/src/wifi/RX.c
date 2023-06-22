#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "CyclicPrefix.h"
#include "Preamble_ST_LG.h"
#include "assert.h"
#include "baseband_lib.h"
#include "ch_Est_Equa.h"
#include "channel_Eq.h"
#include "channel_Eq_initialize.h"
#include "channel_Eq_terminate.h"
#include "common.h"
#include "datatypeconv.h"
#include "decode.h"
#include "detection.h"
#include "dma.h"
#include "equalizer.h"
#include "fft_hs.h"
#include "fft_hwa.h"
#include "interleaver_deintleaver.h"
#include "pilot.h"
#include "qpsk_Mod_Demod.h"
#include "rfInf.h"
#include "rf_interface.h"
#include "rt_nonfinite.h"
#include "scrambler_descrambler.h"
#include "viterbi.h"

void get_tx_fft_data();
void get_rx_viterbi_data();
void get_rx_demod_data();

#ifdef PAPI
#include <papi.h>
void papi_init();
int elab_papi_end(const char *format, ...);
#endif

#define SEC2NANOSEC 1000000000

//############################
//## FFT
//############################
extern TYPE *udmabuf1_base_addr;
extern int dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int fd_udmabuf1;
extern int fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern TYPE *udmabuf2_base_addr;
extern int dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int fd_udmabuf2;
extern int fft2_control_fd;
extern unsigned int *fft2_control_base_addr;

TYPE *udmabuf1_base_addr;
int dma1_control_fd;
unsigned int *dma1_control_base_addr;
int fd_udmabuf1;
int fft1_control_fd;
unsigned int *fft1_control_base_addr;
TYPE *udmabuf2_base_addr;
int dma2_control_fd;
unsigned int *dma2_control_base_addr;
int fd_udmabuf2;
int fft2_control_fd;
unsigned int *fft2_control_base_addr;

static char attr[1024];
int fft_id;

extern sem_t mutex;
sem_t mutex;

signed char dec_in[OUTPUT_LEN];
unsigned char dec_out[USR_DAT_LEN];
signed char dec_pun_out[OUTPUT_LEN];
signed char deintl_out[OUTPUT_LEN];
double out_real[OUTPUT_LEN];
double out_img[OUTPUT_LEN];
signed char outbit[OUTPUT_LEN];
unsigned char descram[USR_DAT_LEN * SYM_NUM + 1];
int hw_fft_busy;

unsigned int *base_addr;

void *wifi_rx() {
	int fft_a53_count = 0;
	int fft_acc_count = 0;

	float FFT_WAIT_PROBABILITY = 0.5;
	float FFT_RUN_ON_CPU_PROBABILITY = 0.5;

	FFT_WAIT_PROBABILITY = 0.5;
	FFT_RUN_ON_CPU_PROBABILITY = 0.5;

	int j;

	int decoderId;
	comp_t out_fd[FFT_N];
	comp_t pilot_rm[INPUT_LEN];
	float *offt;
	float pilotdata_rx[PILOT_LEN];
	float out_fft_c2f[INPUT_LEN * 2];
	int frameCount = 0;
	FILE *fp;
	struct timespec start1, end1;
	float exec_time;

	// create_frameDetection(OSRATE); // up sample rate=4
	//#ifdef TARGET
	//    create_rfInf(RXMODE, RFCARD, INT_2BYTE, NO_SMPERR, 0);
	//#else
	//    create_rfInf(RXMODE, DATFILE, INT_2BYTE, NO_SMPERR, 0);
	//#endif

	init_viterbiDecoder();
	decoderId = get_viterbiDecoder();
	set_viterbiDecoder(decoderId);

	// init_equalizer();

	int frame_count = 0;

#ifdef THREAD_PER_TASK
	pthread_t thread_payload;
	pthread_t thread_fft;
	pthread_t thread_pilotex;
	pthread_t thread_pilotrm;
	pthread_t thread_qpsk_demod;
	pthread_t thread_deinterleaver;
	pthread_t thread_decoder;
	pthread_t thread_descrambler;

	pthread_attr_t attr_thread_payload;
	pthread_attr_t attr_thread_fft;
	pthread_attr_t attr_thread_pilotex;
	pthread_attr_t attr_thread_pilotrm;
	pthread_attr_t attr_thread_qpsk_demod;
	pthread_attr_t attr_thread_deinterleaver;
	pthread_attr_t attr_thread_decoder;
	pthread_attr_t attr_thread_descrambler;

	pthread_attr_init(&attr_thread_payload);
	pthread_attr_init(&attr_thread_fft);
	pthread_attr_init(&attr_thread_pilotex);
	pthread_attr_init(&attr_thread_pilotrm);
	pthread_attr_init(&attr_thread_qpsk_demod);
	pthread_attr_init(&attr_thread_deinterleaver);
	pthread_attr_init(&attr_thread_decoder);
	pthread_attr_init(&attr_thread_descrambler);

	pthread_attr_setname(&attr_thread_payload, "payload");
	pthread_attr_setname(&attr_thread_fft, "fft");
	pthread_attr_setname(&attr_thread_pilotex, "pilotex");
	pthread_attr_setname(&attr_thread_pilotrm, "pilotrm");
	pthread_attr_setname(&attr_thread_qpsk_demod, "qpsk_demod");
	pthread_attr_setname(&attr_thread_deinterleaver, "deinterleaver");
	pthread_attr_setname(&attr_thread_decoder, "decoder");
	pthread_attr_setname(&attr_thread_descrambler, "descrambler");
#endif

	while (frame_count < NUM_FRAMES) {
		//............... Frame Detection ........................
		if (frameDetection() == 0) frameDetection();

		// clean RX buffer
		for (j = 0; j < SYM_NUM * USR_DAT_LEN; j++) descram[j] = 0;

		for (j = 0; j < SYM_NUM; j++) {
#ifdef ACC_RX_FFT
			get_tx_fft_data();
#endif

#ifdef ACC_RX_DEMOD
			get_rx_demod_data();
#endif

#ifdef ACC_RX_DECODER
			get_rx_viterbi_data();
#endif

			// srand(clock());
			srand(frame_count + 1);
#ifndef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

//###############################################################
//## Payload Extraction
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			payloadExt(out_fd);
#else
			struct args_payload *thread_param_payload = (struct args_payload *)malloc(sizeof(struct args_payload));
			thread_param_payload->dbuf = out_fd;
			assert(pthread_create(&thread_payload, &attr_thread_payload, payloadExt, (void *)thread_param_payload) ==
			       0);
			assert(pthread_join(thread_payload, NULL) == 0);
			free(thread_param_payload);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-PayloadExt execution time (ns): %f\n", exec_time);
#endif

			//###############################################################
			//## FFT
			//###############################################################
			int random_wait_time_in_us;  // = 15;
			float random_wait;
			float random_wait1;
			float random_wait2;
			float run_on_cpu;

			fft_id = 1;

#ifdef FFT_HW
			hw_fft_busy = rand() % 2;

#ifdef FFT1_HW_ONLY
			hw_fft_busy = 0;
			fft_id = 1;
#endif

#ifdef FFT2_HW_ONLY
			hw_fft_busy = 0;
			fft_id = 2;
#endif

#if !defined(FFT1_HW_ONLY) && !defined(FFT2_HW_ONLY)

#ifdef FFT_HW_ONLY
#ifdef FFT_SWITCH_INST_IF_BUSY
			fft_id = (hw_fft_busy == 1) ? 2 : 1;
			hw_fft_busy = 0;
			random_wait = rand() / (float)RAND_MAX;
			if (random_wait < FFT_WAIT_PROBABILITY) {
				while (1) {
					random_wait_time_in_us = rand() % 20;
					random_wait_time(random_wait_time_in_us);
					random_wait1 = rand() / (float)RAND_MAX;
					if (random_wait1 > FFT_WAIT_PROBABILITY) {
						fft_id = 1;
						break;
					}
					random_wait2 = rand() / (float)RAND_MAX;
					if (random_wait2 > FFT_WAIT_PROBABILITY * FFT_WAIT_PROBABILITY) {
						fft_id = 2;
						break;
					}
				}
			}
#else
			fft_id = 1;
			random_wait = rand() / (float)RAND_MAX;
			while (random_wait < FFT_WAIT_PROBABILITY) {
				random_wait_time_in_us = rand() % 20;
				random_wait_time(random_wait_time_in_us);
				random_wait = rand() / (float)RAND_MAX;
			}
#endif

			hw_fft_busy = 0;
#else
			run_on_cpu = rand() / (float)RAND_MAX;
			if (run_on_cpu > FFT_RUN_ON_CPU_PROBABILITY) {
				random_wait = rand() / (float)RAND_MAX;
				while (random_wait < FFT_WAIT_PROBABILITY) {
					random_wait_time_in_us = rand() % 20;
					random_wait_time(random_wait_time_in_us);
					random_wait = rand() / (float)RAND_MAX;
				}
				hw_fft_busy = 0;
			} else {
				hw_fft_busy = 1;
			}
#endif

#endif

			if (hw_fft_busy == 1) fft_a53_count++;
			if (hw_fft_busy == 0) fft_acc_count++;

#else

			hw_fft_busy = 1;
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#endif

			if (sem_wait(&mutex) != 0) {
				printf("[ERROR] Semaphore wait failed ...\n");
				exit(-1);
			}

#ifndef THREAD_PER_TASK
			fft_hs(fft_id, out_fd, FFT_N, hw_fft_busy);
#else
			struct args_fft *thread_param_fft = (struct args_fft *)malloc(sizeof(struct args_fft));
			thread_param_fft->fft_id = fft_id;
			thread_param_fft->fdata = out_fd;
			thread_param_fft->n = FFT_N;
			thread_param_fft->hw_fft_busy = hw_fft_busy;
			assert(pthread_create(&thread_fft, &attr_thread_fft, fft_hs, (void *)thread_param_fft) == 0);
			assert(pthread_join(thread_fft, NULL) == 0);
			free(thread_param_fft);
#endif

			if (sem_post(&mutex) != 0) {
				printf("[ERROR] Semaphore post failed ...\n");
				exit(-1);
			}

#ifndef FFT_HW

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-FFT execution time (ns): %f\n", exec_time);
#endif

#endif

//###############################################################
//## Pilot Extraction
//###############################################################
//#ifdef PRINT_BLOCK_EXECUTION_TIMES
// clock_gettime(CLOCK_MONOTONIC, &start1);
//#endif

//#ifndef THREAD_PER_TASK
// pilotExtract(out_fd, pilotdata_rx);
//#else
// struct args_pilotex *thread_param_pilotex = (struct args_pilotex *)malloc(sizeof(struct args_pilotex));
// thread_param_pilotex->idata = out_fd;
// thread_param_pilotex->pilot_data = pilotdata_rx;
// assert(pthread_create(&thread_pilotex, &attr_thread_pilotex, pilotExtract, (void *)thread_param_pilotex) == 0);
// assert(pthread_join(thread_pilotex, NULL) == 0);
// free(thread_param_pilotex);
//#endif

//#ifdef PRINT_BLOCK_EXECUTION_TIMES
// clock_gettime(CLOCK_MONOTONIC, &end1);
// exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC +
// (double)start1.tv_nsec); printf("[INFO] RX-PilotExt execution time (ns): %f\n", exec_time); #endif

//....... Channel Estimation and Equalization ............
// if(j == 0) equalization(pilotdata_rx, offt, out_fft_c2f, 0); // estimation
// equalization(pilotdata_rx, offt, out_fft_c2f, 1); // equalization

//###############################################################
//## Pilot Removal
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			pilotRemove(FFT_N, out_fd, pilot_rm);
#else
			struct args_pilotrm *thread_param_pilotrm = (struct args_pilotrm *)malloc(sizeof(struct args_pilotrm));
			thread_param_pilotrm->len = FFT_N;
			thread_param_pilotrm->idata = out_fd;
			thread_param_pilotrm->odata = pilot_rm;
			assert(pthread_create(&thread_pilotrm, &attr_thread_pilotrm, pilotRemove, (void *)thread_param_pilotrm) ==
			       0);
			assert(pthread_join(thread_pilotrm, NULL) == 0);
			free(thread_param_pilotrm);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-PilotRmv execution time (ns): %f\n", exec_time);
#endif

			for (int i = 0; i < INPUT_LEN; i++) {
				out_fft_c2f[2 * i] = pilot_rm[i].real;
				out_fft_c2f[2 * i + 1] = pilot_rm[i].imag;
			}

			//###############################################################
			//## QPSK Demodulation
			//###############################################################

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			DeMOD_QPSK(INPUT_LEN, pilot_rm, outbit);
#else
			struct args_qpsk_demod *thread_param_qpsk_demod =
			    (struct args_qpsk_demod *)malloc(sizeof(struct args_qpsk_demod));
			thread_param_qpsk_demod->n = INPUT_LEN;
			thread_param_qpsk_demod->ibuf = pilot_rm;
			thread_param_qpsk_demod->out = outbit;
			assert(pthread_create(&thread_qpsk_demod, &attr_thread_qpsk_demod, DeMOD_QPSK,
			                      (void *)thread_param_qpsk_demod) == 0);
			assert(pthread_join(thread_qpsk_demod, NULL) == 0);
			free(thread_param_qpsk_demod);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-QPSK execution time (ns): %f\n", exec_time);
#endif

//###############################################################
//## Deinterleaver
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			deinterleaver(outbit, OUTPUT_LEN, deintl_out);
#else
			struct args_deinterleaver *thread_param_deinterleaver =
			    (struct args_deinterleaver *)malloc(sizeof(struct args_deinterleaver));
			thread_param_deinterleaver->datain = outbit;
			thread_param_deinterleaver->N = OUTPUT_LEN;
			thread_param_deinterleaver->top2 = deintl_out;
			assert(pthread_create(&thread_deinterleaver, &attr_thread_deinterleaver, deinterleaver,
			                      (void *)thread_param_deinterleaver) == 0);
			assert(pthread_join(thread_deinterleaver, NULL) == 0);
			free(thread_param_deinterleaver);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-Deinterleaver execution time (ns): %f\n", exec_time);
#endif

////format conversion
#ifdef HARDINPUT

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

			formatConversion(PUNC_RATE_1_2, deintl_out, dec_in);

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-FmtConversion execution time (ns): %f\n", exec_time);
#endif

#endif
//          // depuncturing
//#ifdef HARDINPUT
//          viterbi_depuncturing(PUNC_RATE_1_2, dec_in, dec_pun_out);
//#else
//          viterbi_depuncturing(PUNC_RATE_1_2, deintl_out, dec_pun_out);
//#endif

//###############################################################
//## Viterbi Decoder
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			viterbi_decoding(decoderId, dec_in, dec_out);
#else
			struct args_decoder *thread_param_decoder = (struct args_decoder *)malloc(sizeof(struct args_decoder));
			thread_param_decoder->dId = decoderId;
			thread_param_decoder->iBuf = dec_in;
			thread_param_decoder->oBuf = dec_out;
			assert(pthread_create(&thread_decoder, &attr_thread_decoder, viterbi_decoding,
			                      (void *)thread_param_decoder) == 0);
			assert(pthread_join(thread_decoder, NULL) == 0);
			free(thread_param_decoder);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-Decoder execution time (ns): %f\n", exec_time);
#endif

//###############################################################
//## Descrambler
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

#ifndef THREAD_PER_TASK
			descrambler(USR_DAT_LEN, dec_out, &descram[j * USR_DAT_LEN]);
#else
			struct args_descrambler *thread_param_descrambler =
			    (struct args_descrambler *)malloc(sizeof(struct args_descrambler));
			thread_param_descrambler->inlen = USR_DAT_LEN;
			thread_param_descrambler->ibuf = dec_out;
			thread_param_descrambler->obuf = &descram[j * USR_DAT_LEN];
			assert(pthread_create(&thread_descrambler, &attr_thread_descrambler, descrambler,
			                      (void *)thread_param_descrambler) == 0);
			assert(pthread_join(thread_descrambler, NULL) == 0);
			free(thread_param_decoder);
#endif

#ifdef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[INFO] RX-Decrambler execution time (ns): %f\n\n", exec_time);
#endif

#ifndef PAPI
#ifndef PRINT_BLOCK_EXECUTION_TIMES
			clock_gettime(CLOCK_MONOTONIC, &end1);
			exec_time = ((double)end1.tv_sec * SEC2NANOSEC + (double)end1.tv_nsec) -
			            ((double)start1.tv_sec * SEC2NANOSEC + (double)start1.tv_nsec);
			printf("[ INFO] RX-chain execution time (ns): %f\n", exec_time);
#endif
#endif
		}

//###############################################################
//## Message Decoder
//###############################################################
#ifdef PRINT_BLOCK_EXECUTION_TIMES
		clock_gettime(CLOCK_MONOTONIC, &start1);
#endif

		messagedecoder((unsigned char *)descram);

		//#ifdef PRINT_BLOCK_EXECUTION_TIMES
		// clock_gettime(CLOCK_MONOTONIC, &end1);
		// exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC +
		// (double)start1.tv_nsec); printf("\n[INFO] RX-MsgDecode execution time (ns): %f\n", exec_time); #endif

		//#ifndef PRINT_BLOCK_EXECUTION_TIMES
		// clock_gettime(CLOCK_MONOTONIC, &end1);
		// exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC +
		// (double)start1.tv_nsec); printf("[INFO] RX-chain execution time (ns): %f\n", exec_time); #endif

		frame_count++;
	}

#ifdef FFT_HW
	printf("\n[INFO] Count of FFT on A53: %d\n", fft_a53_count);
	printf("[INFO] Count of FFT on ACC: %d\n\n\n", fft_acc_count);
#endif
}
