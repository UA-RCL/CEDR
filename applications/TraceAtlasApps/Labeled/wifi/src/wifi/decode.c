#include <stdio.h>
#include <string.h>
#include "common.h"
#include "decode.h" 
#include "crc.h"
#include <stdlib.h>

unsigned char myId =0;

void messagedecoder(unsigned char *buf) {

   int i, j;
   unsigned char ch;
   unsigned char rxBuf[1024];
   int dataLen = (SYM_NUM * USR_DAT_LEN) / 8  ;
   unsigned short rx_crc;
   static unsigned char seqNum = 0xFF;

   // reconstruct byte stream
   for(i=0; i<dataLen; i++) {
      ch = 0;
      for(j=0; j<8; j++) {
         ch <<= 1;
         ch += buf[i*8+j];
      }
      rxBuf[i] = ch;
   }

   // crc check
   rx_crc = crc_ccitt(rxBuf, dataLen);
   if(rx_crc != GOOD_CRC) {
      printf("CRC error !!\n");
      return;
   }

   // terminal ID check
   if(rxBuf[0] != myId) {
      printf("not my packet !!\n");
      return;
   }

   // sequence ID check
   if(seqNum == rxBuf[1]) {
      printf("reception of duplicated packet\n");
//exit(1);
      return;
   }
   else seqNum = rxBuf[1];

   #ifndef PAPI
   // dump RX data
   printf("\n===========================================================\n");
   printf("terminal ID = %X, ", rxBuf[0]);
   printf("seq number = %d\n", rxBuf[1]);
/*
   printf("payload = \n");
   for(i=2; i<(SYM_NUM*5-2); i++) {
      printf("%02X ", rxBuf[i]);
      if((i-2)%20 == 19) printf("\n");
   }
   printf("\n");
*/
   
   printf("Rx string = \n");
   for(i=2; i<(SYM_NUM * SYM_BYTE_LEN)-2; i++) {
      printf("%c ", rxBuf[i]);
      if((i-2)%30 == 29) printf("\n");
   }
   printf("\n");
   #endif
}
