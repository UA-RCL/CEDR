#ifndef _VITERBI_H_
#define _VITERBI_H_

/**
 * Necessary structure for pthreads
 * when sending arguments to the 
 * child thread
 */
struct args_encoder {
  int eId;
  unsigned char *iBuf;
  signed char *oBuf;
};

struct args_decoder {
  int dId;
  unsigned char *iBuf;
  signed char *oBuf;
};

void init_viterbiEncoder();
void init_viterbiDecoder();

int get_viterbiEncoder();
int get_viterbiDecoder();

void set_viterbiEncoder(int eId);
void set_viterbiDecoder(int dId);

#ifndef THREAD_PER_TASK
void viterbi_encoding(int eId, unsigned char *iBuf, signed char *oBuf);
#else
void* viterbi_encoding(void *input);
#endif

#ifndef THREAD_PER_TASK
void viterbi_decoding(int dId, signed char *iBuf, unsigned char *oBuf);
#else
void* viterbi_decoding(void* input);
#endif

void viterbi_puncturing(char rate, signed char *iBuf, signed char *oBuf);
void viterbi_depuncturing(char rate, signed char *iBuf, signed char *oBuf);

void formatConversion(char rate, signed char *iBuf, signed char *oBuf);

#define K            7     // constraits length

// for encoder : 1 represent XOR operation
#define MASK0        0133  // 1011011
#define MASK1        0171  // 1111001 
#define MASK2        0165  // 1110101 

// puncturing
#define PUNC_RATE_1_2  1
#define PUNC_RATE_2_3  2
#define PUNC_RATE_1_3  3



#define SM_STEP      USR_DAT_LEN    // number of column in survivor memory. must be > K*5
#define SM_STATE     64   // number of low in survivor memory. 2^(K-1)
#define IN_BRANCH    2     // number of input branch in trellis diagram. 2^IN_RATE

typedef struct {
  enum entity_state state;
} viterbiEncoder_t;

struct survivorMemory_t {
  int step, state;
  // previous node information;
  unsigned char branchCount;                  // ??
  struct survivorMemory_t *prev[IN_BRANCH];   // pointer to states in previous trellis column
  unsigned char input[IN_BRANCH];             // corresponding input data
  signed char output[IN_BRANCH][OUT_RATE];           // output data of code sequence
  int cost[IN_BRANCH];                        // branch cost
  unsigned char minPathId;                    // input path that has minimum cost.
  int minCost;                                // the cost of selected path
};

typedef struct {
  enum entity_state state;
  struct survivorMemory_t smu[SM_STEP][SM_STATE];
  int init_cost[SM_STATE];
  int initFlag;
} viterbiDecoder_t;

#endif
