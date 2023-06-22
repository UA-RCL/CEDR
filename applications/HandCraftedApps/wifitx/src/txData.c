#include <stdio.h>
#include <stdlib.h>
#include "crc.h"
#include "common.h"



int txDataGen(int option, unsigned char inbit[], int symNum) {

   int i, j, k;
   int dataLen = SYM_BIT_LEN*symNum;
   int payloadLen = dataLen/8 - 2;
   unsigned char txBuf[1024];
   char ch;
   unsigned short tx_crc;
   static unsigned char seqNum = 0;

   for(i=0; i<1024; i++) txBuf[i] = 0;

   // destination terminal ID
   txBuf[0] = 0;

   // message sequence number
   txBuf[1] = seqNum++;

   // 96 byte message
   i = 2;

   if(option == 0) {
      for(j=0; j<26; j++) txBuf[i++] = 'a' + j;
      for(j=0; j<26; j++) txBuf[i++] = 'A' + j;
      for(j=0; j<10; j++) txBuf[i++] = '0' + j;
      for(j=0; j<26; j++) txBuf[i++] = 'a' + j;
      for(j=0; j<26; j++) txBuf[i++] = 'A' + j;
      for(j=0; j<10; j++) txBuf[i++] = '0' + j;
   }
   else {
      printf("=========================================================\n");
      printf("TX string: ");
      //gets((char *)&txBuf[2]);
      printf("\n");
   }

   // CRC insertion
   tx_crc = crc_ccitt(txBuf, payloadLen);
   tx_crc ^= 0xffff;
   txBuf[payloadLen] = tx_crc & 0x00ff;
   txBuf[payloadLen+1] = (tx_crc >> 8) & 0x00ff; 

   // byte to bit conversion
   for(i=0; i<symNum; i++) {
      for(j=0; j<SYM_BYTE_LEN; j++) {
         ch = txBuf[i*SYM_BYTE_LEN+j];
         for(k=7; k>=0; k--) {
            inbit[((i*SYM_BYTE_LEN+j)*8)+7-k] = (ch >> k) & 0x1;
         }
      }
   }

   if(option == 0) {
      printf("\n================================================\n");
      printf("Dest Id = %d, ", txBuf[0]);
      printf("Seq. Number = %d\n", txBuf[1]);
      for(i=2; i<(symNum*SYM_BYTE_LEN)-2; i++) {
         printf("%c ", txBuf[i]);
         if((i-2)%30 == 29) printf("\n");
      }
      printf("\n");
   }
   
   return dataLen;
}
