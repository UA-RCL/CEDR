#ifndef _COMMON_H_
#define _COMMON_H_

#define SYM_NUM      	10
#define USR_DAT_LEN 	64
#define SYM_BYTE_LEN 	8
#define SYM_BIT_LEN (SYM_BYTE_LEN*8)
enum entity_state { IDLE, BUSY };
//viterbi
#define MAX_VITERBI     5
//#define CODE_BLOCK      56   // TB:148, CRC:16
#define CODE_BLOCK      USR_DAT_LEN
#define IN_RATE     	 1     // input per cycle
#define OUT_RATE   		 2     // output per cycle
#define ZERO_PADDING    48     // zero pading for trellis termination
#define INPUT_LEN       (USR_DAT_LEN + ZERO_PADDING)       // input buffer size, unit is bit
#define INPUT_LEN_BYTE  (INPUT_LEN/8 + 1)                 // input buffer size, unit is byte
#define OUTPUT_LEN      (INPUT_LEN * OUT_RATE / IN_RATE)  // output buffer size, unit is bit
#define OUTPUT_LEN_BYTE (OUTPUT_LEN/8)                // output buffer size, unit is byte

//#define Nbits 40
//#define q_len 			Nbits/2 
#define FFT_N  			128
#define PREAMBLE_LEN 	322
#define CYC_LEN 		(FFT_N*25/100)
#define PILOT_LEN 		16
#define TOTAL_LEN 		(FFT_N + CYC_LEN)

#define TX_DATA_LEN 	2500

//#define DEBUG 1

#define FILE_TX_RX 1
//#define WIRELESS 1
#define HARDINPUT 1
//#define WHOLE_CHAIN_DEBUG 1
//#define AWGN_channel 1
//#define Rayliegh_channel 1

#define TYPE float
#define DIM  128

//#define PRINT_BLOCK_EXECUTION_TIMES 

#define FFT_WAIT_PROBABILITY 0.5

#define NUM_FRAMES 10
//#define FFT_HW
//#define FFT1_HW
//#define FFT2_HW
//#define FFT_SWITCH_INST_IF_BUSY
//#define FFT_HW_ONLY
//#define FFT1_HW_ONLY
//#define FFT2_HW_ONLY
//#define SCRAMBLER_CE_HW

#endif
