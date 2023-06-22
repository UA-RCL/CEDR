#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include "viterbi.h"
#include <time.h>

#define SEC2NANOSEC 1000000000

viterbiEncoder_t vEncoder[MAX_VITERBI];
viterbiDecoder_t vDecoder[MAX_VITERBI];

void init_viterbiEncoder() {
  
  int i;

  for(i=0; i<MAX_VITERBI; i++) {
    vEncoder[i].state = IDLE;
  }
}

void init_viterbiDecoder() {

  int i;

  for(i=0; i<MAX_VITERBI; i++) {
    vDecoder[i].state = IDLE;
  }
}

int get_viterbiEncoder() {

  int i;

  for(i=0; i<MAX_VITERBI; i++) {
    if(vEncoder[i].state == IDLE) {
      vEncoder[i].state = BUSY;
      return i;
    }
  }

  printf("no more viterbi encoder !!\n");
  exit(EXIT_FAILURE);
}

int get_viterbiDecoder() {

  int i;

  for(i=0; i<MAX_VITERBI; i++) {
    if(vDecoder[i].state == IDLE) {
      vDecoder[i].state = BUSY;
      return i;
    }
  }

  printf("no more viterbi decoder !!\n");
  exit(EXIT_FAILURE);
}

void set_viterbiEncoder(int eId) {

}

void viterbi_puncturing(char rate, signed char *iBuf, signed char *oBuf) {

  int i;
  int obIndex = 0;
  int puncIndex = 0;
  int puncBase = 0;
  int puncConfig[18]  = { 1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  0,  1,  1,  0,  1};
  int puncShuffle[18] = { 0,  2, -1,  4,  6, -1,  8, 10, -1,  1, -1,  3,  5, -1,  7,  9, -1,  11};

  if(rate == PUNC_RATE_1_2 || rate == PUNC_RATE_1_3) {
    for(i=0; i<OUTPUT_LEN; i++) oBuf[i] = iBuf[i];
  }
  else if(rate == PUNC_RATE_2_3) {
    for(i=0; i<OUTPUT_LEN; i++) {
      puncIndex = i%18;
      if(puncIndex == 0) {
        if(i==0) puncBase = i;
        else puncBase += 12;
      }
      if(puncConfig[puncIndex] == 1) {
        obIndex = puncShuffle[puncIndex] + puncBase;
        oBuf[obIndex++] = iBuf[i];
      }
    }
  }
}

void viterbi_depuncturing(char rate, signed char *iBuf, signed char *oBuf) {

  int i;
  int depuncShuffle[12] = { 0,  9,  1, 11,  3, 12,  4, 14,  6, 15,  7, 17};  
  int depuncIndex = 0;
  int depuncBase = 0;
  int obIndex = 0;
  int len;

  if(rate == PUNC_RATE_1_2 || rate == PUNC_RATE_1_3) {
    for(i=0; i<OUTPUT_LEN; i++) oBuf[i] = iBuf[i];
  }
  else if(rate == PUNC_RATE_2_3) {
    len = OUTPUT_LEN * 2 / 3;
    for(i=0; i<OUTPUT_LEN; i++) oBuf[i] = 0; 
    for(i=0; i<len; i++) {
      depuncIndex = i % 12;
      if(depuncIndex == 0) {
        if(i==0) depuncBase = i;
        else depuncBase += 18; 
      }
      obIndex = depuncShuffle[depuncIndex] + depuncBase;
      oBuf[obIndex] = iBuf[i];
    }
  }

}

void bitEncoding(unsigned char state, int input, signed char *out) {

  int i;
  unsigned char g0, g1;

  g0 = g1 = input;

  for(i=0; i<(K-1); i++) {
    if((MASK0 >> i) & 0x1) g0 ^= ((state>>i) & 0x1);
    if((MASK1 >> i) & 0x1) g1 ^= ((state>>i) & 0x1);
  }

  out[0] = g0 ? -15 : 15;
  out[1] = g1 ? -15 : 15;

//printf("state = %d, %d, %d%d%d\n", state, input, g0,g1,g2);
}

void set_viterbiDecoder(int dId) {
  
  int i, j;
  int count, nState;
  viterbiDecoder_t *vd;

  vd = &vDecoder[dId];

  for(i=0; i<SM_STEP; i++) {
    for(j=0; j<SM_STATE; j++) {
      vd->smu[i][j].step = i;
      vd->smu[i][j].state = j;
    }
  }
  
  // initialize survivor memory unit
  // We can calculate the outputs correspdond to state transition
  // before we have input data. It is calculated at initialization
  // phase and resused.
  for(i=0; i<SM_STEP; i++) {
    for(j=0; j<SM_STATE; j++) {
      // input 0 case
      nState = j >> 1;
      count = vd->smu[i][nState].branchCount++;
      if(i==0) vd->smu[i][nState].prev[count] = &(vd->smu[SM_STEP-1][j]);
      else vd->smu[i][nState].prev[count] = &(vd->smu[i-1][j]);
      vd->smu[i][nState].input[count] = 0;
      bitEncoding(j, 0, vd->smu[i][nState].output[count]);
      count++;

      // input 1 case
      nState = SM_STATE/2 + (j >> 1);
      count = vd->smu[i][nState].branchCount++;
      if(i == 0)  vd->smu[i][nState].prev[count] = &(vd->smu[SM_STEP-1][j]);
      else vd->smu[i][nState].prev[count] = &(vd->smu[i-1][j]);
      vd->smu[i][nState].input[count] = 1;
      bitEncoding(j, 1, vd->smu[i][nState].output[count]);
    }
  }
}

void viterbi_encoding(int eId, unsigned char *iBuf, signed char *oBuf) {

  int i, j, k;

  int oIndex;
  unsigned char sReg, iBit, g0, g1;

  // initialization
  for(i=0; i<OUTPUT_LEN; i++) oBuf[i] = 0;
  sReg = 0;

  // zero padding
  for(i=CODE_BLOCK, j=0; j<ZERO_PADDING; i++, j++) iBuf[i] = 0; //changed

  for(i=0, oIndex=0; i<SM_STEP; i++) {

    iBit = iBuf[i];

    for(k=0, g0=iBit; k<(K-1); k++) {
      if((MASK0 >> k & 0x01)) {
	g0 = g0 ^ (sReg >> k & 0x1);
      }
    }

    for(k=0, g1=iBit; k<(K-1); k++) {
      if((MASK1 >> k & 0x01)) {
	g1 = g1 ^ (sReg >> k & 0x1);
      }
    }

    // update register
    sReg = (sReg >> 1) + (iBit << (K-2));

    // fill output buffer
    oBuf[oIndex++] = g0;
    oBuf[oIndex++] = g1;
  }

}

void smu_dump(int dId, int start, int end) {

  int i, j;
  viterbiDecoder_t *vd;
  struct survivorMemory_t *sm;

  vd = &vDecoder[dId];

  printf("\n\t\t");
  for(i=start; i<=end; i++) printf("%d\t\t\t",i);
  printf("\n");

  for(i=0; i<SM_STATE; i++) {
    printf("%d:minCost", i);
    for(j=start; j<=end; j++) {
      sm = &vd->smu[j][i];
      printf("\t%d\t\t", sm->minCost);
    }
    printf("\n");
    printf("%d:Input0", i);
    for(j=start; j<=end; j++) {
      sm = &vd->smu[j][i];
      printf("\t%4d,%4d:%4d,", sm->output[0][0], sm->output[0][1], sm->cost[0]);
      if(j==0) printf("N");
      else printf("%d", sm->prev[0]->state);
    }
    printf("\n");
    printf("%d:Input1", i);
    for(j=start; j<=end; j++) {
      sm = &vd->smu[j][i];
      printf("\t%4d,%4d:%4d,", sm->output[1][0], sm->output[1][1], sm->cost[1]);
      if(j==0) printf("N");
      else printf("%d", sm->prev[1]->state);
    }
    printf("\n");
  }
}

void viterbi_initSmu(int dId, int column, int initFlag) {

  int i, j;
  viterbiDecoder_t *vd;

  vd = &vDecoder[dId];

  for(i=0; i<SM_STATE; i++) {
    for(j=0; j<2; j++) vd->smu[column][i].cost[j] = 0;
    if(initFlag) {
      vd->initFlag = 1;
      vd->smu[column][i].minCost = vd->init_cost[i]; // load initial cost
    }
    else vd->smu[column][i].minCost = 0;
  }
}

void viterbi_BMC(int dId, int column, signed char *iBuf) {

  int i, j, k;
  struct survivorMemory_t *sm;

  for(i=0; i<SM_STATE; i++) {
    sm = &vDecoder[dId].smu[column][i];
    for(j=0; j<2; j++) {
      // calculate euclidian distance
      for(k=0; k<2; k++) {
        if(sm->output[j][k] > iBuf[k]) { // input 0
	  sm->cost[j] += (sm->output[j][k] - iBuf[k]);
	}
	else { // input 1
	  sm->cost[j] += (iBuf[k] - sm->output[j][k]);
	}
      }
    }
  }
}

void viterbi_ACS(int dId, int column) {

  int i, j;
  viterbiDecoder_t *vd;
  struct survivorMemory_t *sm;

  vd = &vDecoder[dId];

  for(i=0; i<SM_STATE; i++) {
    sm = &vd->smu[column][i];
    //add
    if(!vd->initFlag) {
      for(j=0; j<2; j++) sm->cost[j] += sm->prev[j]->minCost;
    }
    // compare and select
    if(sm->cost[0] > sm->cost[1]) {
      sm->minPathId = 1;
      sm->minCost = sm->cost[1];
    }
    else {
      sm->minPathId = 0;
      sm->minCost = sm->cost[0];
    }
  }

  vd->initFlag = 0;
}

void viterbi_TB(int dId, int oIndex, unsigned char *oBuf) {

  int i;
  viterbiDecoder_t *vd;
  struct survivorMemory_t *sm;
  short minCost, minId;

  vd = &vDecoder[dId];
  
  // Search at last column
  minCost = vd->smu[oIndex][0].minCost;
  minId = 0;
  for(i=1; i<SM_STATE; i++) {
    if(vd->smu[oIndex][i].minCost < minCost) {
      minCost = vd->smu[oIndex][i].minCost;
      minId = i;
    }
  }

  // traceback inputs
  sm = &vd->smu[oIndex][minId];
  for(i=oIndex; i>=0; i--) {
    oBuf[i] = sm->input[sm->minPathId];
    sm = sm->prev[sm->minPathId];
  }
}

//********************************************
// called in every TTI
//********************************************
void viterbi_decoding(int dId, signed char *iBuf, unsigned char *oBuf) {

  int i, column;
  viterbiDecoder_t *vd;

  vd = &vDecoder[dId];

  //unsigned int BASE = 0x6d800000;
  //for(i=0; i<128; i++) printf("devmem 0x%x 8 0x%02x\n", (BASE + i), iBuf[i]);

  // initialize output buffer
  for(i=0; i<USR_DAT_LEN; i++) oBuf[i] = 0;

  // set initial path cost of the states in first column
  // Because encoder always begins with zero state, set minimum
  // cost (zero) on state 00000000. Other states are set by 0xFFFF
  vd->init_cost[0] = 0;
  for(i=1; i<SM_STATE; i++) vd->init_cost[i] = 0xFFFF;

  for(i=0; i<SM_STEP; i++) {
    column = i;
    if(column==0) viterbi_initSmu(dId, column, 1);
    else viterbi_initSmu(dId, column, 0);
    viterbi_BMC(dId, column, &iBuf[i*OUT_RATE]);
    viterbi_ACS(dId, column);
  }

//  smu_dump(dId, 0, 3);

  viterbi_TB(dId, SM_STEP-1, oBuf);

  //for(i = 0; i < 64; i++) printf("Output [%d] - 0x%x\n", i, oBuf[i]);
}


//********************************************
// Format Conversion
//********************************************

void formatConversion(char rate, signed char *iBuf, signed char *oBuf) {
    
    int i,k;
    //unsigned char data;
    int len;
    int ten = 10, minusTen = -10;
    
    len = (rate == PUNC_RATE_1_2) ? OUTPUT_LEN : (OUTPUT_LEN * 2 / 3);
    
    k=0;
    for(i=0; i<len; i++) {
        if(iBuf[i] == 1) oBuf[k++] = minusTen;
        else oBuf[k++] = ten;
        
    }
}

