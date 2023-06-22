#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "baseband_lib.h"
#include "rfInf.h"

//#define BUF_LIMIT (TRF_UNIT * 2)
#define BUF_LIMIT 1000000
#define SEC2NANOSEC 1000000000

static comp_t rfIoBuf[BUF_LIMIT];

static FILE *lfp1, *lfp2, *lfp3, *lfp4, *lfp5;
static float mBuf[6];
static long int sampleCount = 0;
static int maxIndex = 0; 
static long int prtMaxIdx = 0;
static int payloadStart = 0, payloadPrtStart = 0;
//static int spRate = OSR_3;
static int spRate = 1;
static int sIdx = 0;

static comp_t longPreamble[128] = {
   {-0.156,0.000},  { 0.012,-0.098}, { 0.092,-0.106}, {-0.092,-0.115}, {-0.003,-0.054}, { 0.075, 0.074}, {-0.127,0.021},  {-0.122, 0.017}, 
   {-0.035,0.151},  {-0.056,0.022},  {-0.060,-0.081}, { 0.070,-0.014}, { 0.082,-0.092}, {-0.131,-0.065}, {-0.057,-0.039}, { 0.037,-0.098},  
   { 0.062,0.062},  { 0.119,0.004},  {-0.022,-0.161}, { 0.059,0.015},  { 0.024,0.059},  {-0.137,0.047},  { 0.001,0.115},  { 0.053,-0.004},  
   { 0.098,0.026},  {-0.038,0.106},  {-0.115,0.055},  { 0.060,0.088},  { 0.021,-0.028}, { 0.097,-0.083}, { 0.040,0.111},  {-0.005,0.120},
   { 0.156,0.000},  {-0.005,-0.120}, { 0.040,-0.111}, { 0.097,0.083},  { 0.021,0.028},  { 0.060,-0.088}, {-0.115,-0.055}, {-0.038,-0.106},
   { 0.098,-0.026}, { 0.053,0.004},  { 0.001,-0.115}, {-0.137,-0.047}, { 0.024,-0.059}, { 0.059,-0.015}, {-0.022,0.161},  { 0.119,-0.004},
   { 0.062,-0.062}, { 0.037,0.098},  {-0.057,0.039},  {-0.131,0.065},  { 0.082,0.092},  { 0.070,0.014},  {-0.060,0.081},  {-0.056,-0.022},
   {-0.035,-0.151}, {-0.122,-0.017}, {-0.127,-0.021}, { 0.075,-0.074}, {-0.003,0.054},  {-0.092,0.115},  { 0.092,0.106},  { 0.012,0.098},
   {-0.156,0.000},  { 0.012,-0.098}, { 0.092,-0.106}, {-0.092,-0.115}, {-0.003,-0.054}, { 0.075, 0.074}, {-0.127,0.021},  {-0.122, 0.017}, 
   {-0.035,0.151},  {-0.056,0.022},  {-0.060,-0.081}, { 0.070,-0.014}, { 0.082,-0.092}, {-0.131,-0.065}, {-0.057,-0.039}, { 0.037,-0.098},  
   { 0.062,0.062},  { 0.119,0.004},  {-0.022,-0.161}, { 0.059,0.015},  { 0.024,0.059},  {-0.137,0.047},  { 0.001,0.115},  { 0.053,-0.004},  
   { 0.098,0.026},  {-0.038,0.106},  {-0.115,0.055},  { 0.060,0.088},  { 0.021,-0.028}, { 0.097,-0.083}, { 0.040,0.111},  {-0.005,0.120},
   { 0.156,0.000},  {-0.005,-0.120}, { 0.040,-0.111}, { 0.097,0.083},  { 0.021,0.028},  { 0.060,-0.088}, {-0.115,-0.055}, {-0.038,-0.106},
   { 0.098,-0.026}, { 0.053,0.004},  { 0.001,-0.115}, {-0.137,-0.047}, { 0.024,-0.059}, { 0.059,-0.015}, {-0.022,0.161},  { 0.119,-0.004},
   { 0.062,-0.062}, { 0.037,0.098},  {-0.057,0.039},  {-0.131,0.065},  { 0.082,0.092},  { 0.070,0.014},  {-0.060,0.081},  {-0.056,-0.022},
   {-0.035,-0.151}, {-0.122,-0.017}, {-0.127,-0.021}, { 0.075,-0.074}, {-0.003,0.054},  {-0.092,0.115},  { 0.092,0.106},  { 0.012,0.098}
};

static comp_t shortPreamble[16] = {
     {0.045999,0.045999}, {-0.132444, 0.002340}, {-0.013473,-0.078525}, {0.142755,-0.012651}, 
     {0.091998,0.000000}, { 0.142755,-0.012651}, {-0.013473,-0.078525}, {-0.132444,0.002340}, 
     {0.045999,0.045999}, { 0.002340,-0.132444}, {-0.078525,-0.013473}, {-0.012651,0.142755}, 
     {0.000000,0.091998}, {-0.012651, 0.142755}, {-0.078525,-0.013473}, {0.002340,-0.132444}
};

float matchedFiltering(comp_t *buf, int rIdx, comp_t *coef, int nTap) {

   int i;
   float cor = 0.0;
   float real, imag;
   float realAcc = 0.0, imagAcc = 0.0;
   float aErg = 0.0;
   float bErg = 0.0;
   float r;
   float temp;
   int idx1;
   int step = spRate;

   for(i=0; i<nTap; i++) {
      idx1 = (rIdx + (i * step)) % BUF_LIMIT;
      real = compMultReal(buf[idx1].real, buf[idx1].imag, coef[i].real, -coef[i].imag);
      imag = compMultImag(buf[idx1].real, buf[idx1].imag, coef[i].real, -coef[i].imag);
      realAcc = realAcc + real;
      imagAcc = imagAcc + imag;
      aErg = aErg + ((buf[idx1].real * buf[idx1].real) + (buf[idx1].imag * buf[idx1].imag));
      bErg = bErg + ((coef[i].real * coef[i].real) + (coef[i].imag * coef[i].imag));
   }
   cor = compMag(realAcc, imagAcc);

   temp = aErg * bErg;

   if(temp < 0.005) r = 0;
   else r = (cor / temp) * 4.0;

//printf("r=%f, cor=%f, aErg=%f, bErg=%f\n", r, cor, aErg, bErg);

   return r;
}


float slidingWindow(comp_t *buf, int rIdx, int wGap, int len, int step, int type) {

   int i;
   float cor = 0.0;
   float real, imag;
   float realAcc = 0.0, imagAcc = 0.0;
   float aErg = 0.0, bErg = 0.0;
   float r;
   float temp;
   int idx1, idx2;

   for(i=0; i<len; i=i+step) {
      idx1 = (rIdx + i) % BUF_LIMIT;
      idx2 = (rIdx + wGap + i) % BUF_LIMIT;
      if(type == 0) {
         real = compMultReal(buf[idx1].real, buf[idx1].imag, buf[idx2].real, buf[idx2].imag);
         imag = compMultImag(buf[idx1].real, buf[idx1].imag, buf[idx2].real, buf[idx2].imag);
      }
      else {
         real = compMultReal(buf[idx1].real, buf[idx1].imag, buf[idx2].real, -buf[idx2].imag);
         imag = compMultImag(buf[idx1].real, buf[idx1].imag, buf[idx2].real, -buf[idx2].imag);
      }
      realAcc = realAcc + real;
      imagAcc = imagAcc + imag;
      aErg = aErg + ((buf[idx1].real * buf[idx1].real) + (buf[idx1].imag * buf[idx1].imag));
      bErg = bErg + ((buf[idx2].real * buf[idx2].real) + (buf[idx2].imag * buf[idx2].imag));
   }
   cor = compMag(realAcc, imagAcc);
   temp = aErg * bErg;

   if(temp < 0.005) r = 0;
   else r = (cor / temp);

   return r;
}

void init_frameDetection() {

   int i;
   for(i=0; i<6; i++) mBuf[i] = 0.0;
}

void create_frameDetection(int uspRate) {

   lfp1 = fopen("../log/idata.log", "w");
   lfp2 = fopen("../log/cor.log", "w");
   lfp3 = fopen("../log/long.log", "w");
   lfp4 = fopen("../log/payload.log", "w");
   lfp5 = fopen("../log/idata_mat.log", "w");

   sampleCount = 0;
   spRate = uspRate;

   init_frameDetection();
}

void delete_frameDetection() {

   fclose(lfp1); 
   fclose(lfp2); 
   fclose(lfp3);
   fclose(lfp4);
   fclose(lfp5);
}


int frameDetection() {

   int k;
   float measure;
   float avg;
   int state = 0;
   int shortDetectCount = 0;
   int shortProveCount = 0;

   float longThreshold = 0.5;
   float shortThreshold = 0.95;
   float max = 0.0;
   int longCount = 0;
   int prtFlag = 0;
   int cpMargin = 32;
   //int ret;
	FILE *cfp, *fp111;
	char buf[81920];
	int i = 0;
sampleCount = 0;
   if(sIdx == 0) {
      init_frameDetection();
      //ret = rfInfRead(&rfIoBuf[0], BUF_LIMIT);
	//cfp = fopen("/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/Applications/wifi/build/txdata_1.txt","r");
	cfp = fopen("./input/txdata_1.txt", "r");
	//fp111 = fopen("rxdata_1.txt","w");
        while(fgets(buf, 81920, cfp) != NULL) {
            sscanf(buf, "%f %f\n", &rfIoBuf[i].real, &rfIoBuf[i].imag);
            ///fprintf(fp111,"%f %f\n", rfIoBuf[i].real, rfIoBuf[i].imag);
            i++;
        }
	fclose(cfp);
	//fclose(fp111);
	}

   for( ; sIdx<BUF_LIMIT; sIdx++) {


//   printf("frame detecting!\n");
      if(state == 0) { // short preamble detection
         measure = slidingWindow(rfIoBuf, sIdx, 16*spRate, 16*spRate, spRate, 1);
         for(k=5; k>0; k--) mBuf[k] = mBuf[k-1];
         mBuf[0] = measure;
         avg = (mBuf[0] + mBuf[1] + mBuf[2] + mBuf[3] + mBuf[4] + mBuf[5]) / 6.0;
//printf("short preamble detection value = %f avg = %f\n", measure, avg);
         if(measure > shortThreshold) {
//printf("short count = %d %d\n",shortDetectCount,10 * spRate);
            if(shortDetectCount++ > (10 * spRate)) {

               prtFlag = 1;

               state = 1; // change to long preamble detection state
               shortProveCount = 0;
//printf("state 1\n");
               for(k=0; k<6; k++) mBuf[k] = 0.0;
            }
         }
         else {
            shortDetectCount = 0;
            if(prtFlag == 1) {
               prtFlag = 0; 

            }
         }
      }
      else if(state == 1) { // matched filtering for short frame detection
         if(shortProveCount++ < (16 *spRate)) {
            measure = matchedFiltering(rfIoBuf, sIdx, shortPreamble, 16);

            if(measure > 3.0) {

               state = 2;
               longCount = 0;
            }
         }
         else {
            state = 0;
            shortDetectCount = 0;
            prtFlag = 0; 

         }
      }
      else if(state == 2) { // long preamble detection
        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        struct timespec start1, end1;
        float exec_time;
        clock_gettime(CLOCK_MONOTONIC, &start1);
        #endif

         if(longCount++ < (150 * spRate)) {
            measure = matchedFiltering(rfIoBuf, sIdx, longPreamble, 64);

            if(measure >  longThreshold) {
               if(max < measure) {

//printf("update max, max = %f, measure = %f at %ld\n", max, measure, sampleCount);

                  max = measure;
                  maxIndex = sIdx;
                  prtMaxIdx = sampleCount;
         #ifdef PRINT_BLOCK_EXECUTION_TIMES
         clock_gettime(CLOCK_MONOTONIC, &end1);
         exec_time = ((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec) - ((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec);
         printf("[INFO] RX-MatchFilter execution time (ns): %f\n", exec_time);
         #endif
  
               }
            }
         }
         else {
            state = 0;
            longCount = 0;
            if(max > 0.0) {
               prtFlag = 0;
               break;

            }
         }
      }
      sampleCount++;
   }

   if(sIdx < (BUF_LIMIT - ((161 + cpMargin + SYM_NUM * (128 + 32)) * spRate))) { 
      payloadStart = (maxIndex + (161 + cpMargin) * spRate);
      payloadPrtStart = prtMaxIdx + (161 + cpMargin) * spRate;

      return 1;
   }
   else {
      sIdx = 0;
      sampleCount = ((sampleCount + TRF_UNIT) / (BUF_LIMIT)) * (BUF_LIMIT);
      return 0;
   }
}

void payloadExt(comp_t dbuf[]) {

   int i, j;
   long int l;

   // copy payload with down sampling
   j=payloadStart;
   l=payloadPrtStart;

   //printf("data payload from %ld, %d\n", l, j);

//printf("payload = \n");
   for(i=0; i<128; i++) {
      dbuf[i] = rfIoBuf[j];
//	printf("%dth: %f + %f\n",j,dbuf[i].real, dbuf[i].imag);

      j = (j+spRate);
      if(j>BUF_LIMIT) break;
      l=l+spRate;
   }

   payloadStart += (TOTAL_LEN) * spRate;
   payloadPrtStart += (TOTAL_LEN) * spRate;
   sIdx = payloadStart;
   sampleCount = payloadPrtStart;
}

