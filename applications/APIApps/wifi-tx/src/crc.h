#ifndef __CRC_H__
#define __CRC_H__

#define CRC_INIT 0xffff
#define GOOD_CRC 0xf0b8
unsigned short crc_ccitt(unsigned char *inbuff, unsigned int len);

#endif
