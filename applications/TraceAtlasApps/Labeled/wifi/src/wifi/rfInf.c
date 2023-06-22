#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "baseband_lib.h"
#include "rfInf.h"
#include "rf_interface.h"

static FILE *fp = NULL;

static int mediaType = DATFILE;
static int dataFormat = INT_2BYTE;
static int correctionType = DN_SMPL;
static int correctionIntvl = UD_INTERVAL;

#ifdef TARGET
extern signed short int rxdata[];
extern signed short int txdata[];
#else
signed short int rxdata[15360*4*100];
signed short int txdata[15360*4*100];
#endif

// len : # of complex elements
// maxLen : # of maximum complex buffer elements
void convIntComp(int len, short int intBuf[], comp_t fBuf[], int maxLen) {

   int i=0, j=0;

   if( len > maxLen ) {
      printf("exceeds buffer limit!!\n");
      exit(1);
   }

   do {
      fBuf[j].real = ((float)intBuf[i++])/1000.0;
      fBuf[j++].imag = ((float)intBuf[i++])/1000.0;
   } while(j<len);
}

// len : # of complex buffer elements
void convCompInt(int len, comp_t fBuf[], short int intBuf[]) {

   int i=0, j=0;
   int udsCount = 0;

   do {
      if(correctionType == NO_SMPERR) { 
         intBuf[i++] = (short int)(fBuf[j].real * 50000);
         intBuf[i++] = (short int)(fBuf[j++].imag * 50000);
      }
      else if(correctionType == DN_SMPL) { 
         if(correctionIntvl > udsCount++) {
            intBuf[i++] = (short int)(fBuf[j].real * 50000);
            intBuf[i++] = (short int)(fBuf[j++].imag * 50000);
         }
         else {
            udsCount = 0;
            j++;
         }
      }
      else {
         printf("Invalid correctionType !!\n");
         exit(1);
      }
   } while(j<len);

//   for( ; i<TRF_UNIT*2; i++) intBuf[i] = 0;
}

static int fileRead(FILE *fp, comp_t *buf, int maxLen) {

   int i;
   int ret;
   comp_t comp;
   struct icomp_str {
      short int real;
      short int imag;
   } icomp;
   static int udsCount = 0; // up/down sampling counter
   int bufCount = 0;

   int sFactor = 5000;

   for(i=0; bufCount<maxLen; i++) {
      if(dataFormat == FLT_4BYTE) { // 4 byte float
         ret = fread(&comp, 4, 2, fp);
         if(ret == 0) return 0;
         *buf = comp;
         buf++;
         bufCount++;
      }
      else if(dataFormat == INT_2BYTE) { // 2 byte integer
         ret = fread(&icomp, 2, 2, fp);
         if(ret == 0) return 0;
         if(correctionType == NO_SMPERR) { // no up/down sampling
            buf->real = (float)icomp.real / sFactor;
            buf->imag = (float)icomp.imag / sFactor;
            buf++;
            bufCount++;
         }
         else if(correctionType == DN_SMPL) { // down sample
            if(correctionIntvl > udsCount++) {
               buf->real = (float)icomp.real / sFactor;
               buf->imag = (float)icomp.imag / sFactor;
               buf++;
               bufCount++;
            }
            else {
               udsCount = 0;
            }
         }
         else if(correctionType == UP_SMPL) { // up sample
            if(correctionIntvl < udsCount++) {
               buf->real = (float)icomp.real / sFactor;
               buf->imag = (float)icomp.imag / sFactor;
               buf++;
               bufCount++;
            }
            else {
               buf->real = (buf-1)->real / sFactor;
               buf->imag = (buf-1)->imag / sFactor;
               buf++;
               bufCount++;
               udsCount = 0;
            }
         }
      }
   }

   return ret;
}

// len : # of complex elements
static int fileWrite(FILE *fp, comp_t *buf, int len) {

   int i, j;
   int ret;
   comp_t comp;
   int bufCount = 0;

   if(dataFormat == FLT_4BYTE) { // 4 byte float
      for(i=0; bufCount<len; i++) {
         comp = buf[i];
         for(j=0; j<OSRATE; j++) {
            ret = fwrite(&comp, 4, 2, fp); // FIXME
            if(ret == 0) return 0;
         }
         bufCount++;
      }
   }
   else if(dataFormat == INT_2BYTE) { // 2 byte integer
      convCompInt(len, buf, txdata);
      for(i=0; bufCount<len; i+=2) {
         for(j=0; j<OSRATE; j++) {
            ret = fwrite(&txdata[i], 2, 2, fp);
            if(ret == 0) return 0;
         }
         bufCount++;
      }
   }

   return ret;
}

int asu_do_one_rx() {
#ifdef TARGET
   int count = 0;
   int i, endFlag = 0, endCount = 0;
   int test_time, rxsze;
   unsigned int output_add;

   test_time = TRF_TIME;
   output_add = (unsigned int)rxdata;
   while(1) {
      asu_radio_init();
/*
      printf("count = %d\n", count++);
*/
      rxsze = asu_tx_rx(0, output_add, test_time);
      for(i=1000; i<5000; i++) {
         if(rxdata[i*2] > 10000) {
//         if(rxdata[i*2] > 2000) {
            if(endCount++ > 50) {
               output_add = (unsigned int)&rxdata[TRF_UNIT];
               rxsze = asu_tx_rx(0, output_add, test_time);
               endFlag = 1;
               break;
            }
         }
      }
      if(endFlag) break;
      else endCount = 0;
   }
   return rxsze*2;
#else
   return 0;
#endif
}

// len : # of short integer elements 
int asu_do_one_tx(int len) {
#ifdef TARGET
   int test_time, ret;
   unsigned int input_add;

   test_time = (len / (15360 * 2)); // ms unit ??
   input_add = (unsigned int)&txdata;
   ret = asu_tx_rx(input_add, 0, test_time);
   return ret;
#else
   return 0;
#endif
}

void create_rfInf(int mode, int media, int format, int crrType, int crrIntvl) {

   mediaType = media;
   dataFormat = format;
   correctionType = crrType;
   correctionIntvl = crrIntvl;

   	switch(mediaType) {
   		case DATFILE :
         	if(mode == RXMODE) 
			{

				// DASH_DATA
				if( !getenv("DASH_DATA") )
				{
					printf("in rfInf.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
					exit(1);
				}

				char* file0 = "Dash-RadioCorpus/wifi/txdata.txt";
				char* path = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
				strcat(path, getenv("DASH_DATA"));
				strcat(path, file0);
				fp = fopen(path, "r");
				free(path);

    			if(fp == NULL) {
        			printf("in rfInf.c:\n\tFATAL: %s was not found!", file0);
        			exit(1);
    			}


			}
     		break;

      case RFCARD :
#ifdef TARGET
         printf("- Using RF ADI card\n");
	 asu_radio_init();
#else
         printf("ERROR : compiled for executing on host machine, but tries to use RFCARD\n");
         exit(1);
#endif
         break;
      default :
         printf("wrong case \n");
         break;
   }
}

void delete_rfInf() {

   switch(mediaType) {
      case DATFILE :
         fclose(fp);
         break;
      case RFCARD :
         //rfInfClose();
         break;
      default :
         printf("wrong case \n");
         break;
   }
}

// maxLen : maximum # of complex elements in buffer
int rfInfRead(comp_t *buf, int maxLen) {

   int ret=0;

   switch(mediaType) {
      case DATFILE :
         ret = fileRead(fp, buf, maxLen);
         break;
      case RFCARD :
         ret = asu_do_one_rx();
//printf("ret = %d, maxLen = %d\n", ret, maxLen);
         convIntComp(ret, rxdata, buf, maxLen);
         break;
      default :
         printf("wrong case \n");
         exit(1);
         break;
   }

   return ret;
}

// len : # of complex elements
int rfInfWrite(comp_t *buf, int len) {

   int ret;
   switch(mediaType) {
      case DATFILE :
         ret = fileWrite(fp, buf, len);
         break;
      case RFCARD :
//exit(1);
         convCompInt(len, buf, txdata);
         ret = asu_do_one_tx(len*2);
         break;
      default :
         printf("wrong case \n");
         exit(1);
         break;
   }

   return ret;
}
