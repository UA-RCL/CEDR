#include "decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "crc.h"

unsigned char myId = 0;

void messagedecoder(unsigned char *buf) {
	int i, j;
	unsigned char ch;
	unsigned char rxBuf[1024];
	int dataLen = (SYM_NUM * USR_DAT_LEN) / 8;
	unsigned short rx_crc;
	static unsigned char seqNum = 0xFF;

	// DASH_DATA
	if (!getenv("DASH_DATA")) {
		printf("in detection.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char *file4 = "Dash-RadioCorpus/QPR8_RFConvSys/txdata_est.txt";
	char *path4 = (char *)malloc(FILEPATH_SIZE * sizeof(char));
	strcat(path4, getenv("DASH_DATA"));
	strcat(path4, file4);
	printf("Opening: %s", path4);
	FILE *txdata_file1 = fopen(path4, "w");
	free(path4);

	if (txdata_file1 == NULL) {
		printf("in decode.c:\n\tFATAL: %s was not found!\n", file4);
		exit(1);
	}

	// reconstruct byte stream
	for (i = 0; i < dataLen; i++) {
		ch = 0;
		for (j = 0; j < 8; j++) {
			ch <<= 1;
			ch += buf[i * 8 + j];
		}
		rxBuf[i] = ch;
	}

	// crc check
	rx_crc = crc_ccitt(rxBuf, dataLen);
	if (rx_crc != GOOD_CRC) {
		printf("CRC error !!\n");
		return;
	}

	// terminal ID check
	if (rxBuf[0] != myId) {
		printf("not my packet !!\n");
		return;
	}

	// sequence ID check
	if (seqNum == rxBuf[1]) {
		printf("reception of duplicated packet\n");
		// exit(1);
		return;
	} else
		seqNum = rxBuf[1];

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

	for (i = 2; i < (SYM_NUM * SYM_BYTE_LEN) - 2; i++) {
		fprintf(txdata_file1, "%c ", rxBuf[i]);
		if ((i - 2) % 30 == 29) fprintf(txdata_file1,"\n");
	}
	fclose(txdata_file1);
	
	printf("Rx string = \n");
	for (i = 2; i < (SYM_NUM * SYM_BYTE_LEN) - 2; i++) {
		printf("%c ", rxBuf[i]);
		if ((i - 2) % 30 == 29) printf("\n");
	}
	printf("\n");
#endif
}
