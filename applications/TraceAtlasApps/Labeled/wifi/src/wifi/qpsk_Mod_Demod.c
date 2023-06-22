#define _GNU_SOURCE
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>
#include "baseband_lib.h"
#include "common.h"
#include "qpsk_Mod_Demod.h"
#include <sched.h>

#define PM1 sqrt(2)/2
#define PM2 -sqrt(2)/2
//#define SOFTINPUT 1

signed char rx_demod_data[128];
unsigned int hex_data;

void get_rx_demod_data() {

hex_data = 0x1; rx_demod_data[0] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[1] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[2] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[3] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[4] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[5] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[6] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[7] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[8] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[9] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[10] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[11] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[12] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[13] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[14] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[15] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[16] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[17] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[18] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[19] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[20] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[21] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[22] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[23] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[24] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[25] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[26] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[27] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[28] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[29] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[30] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[31] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[32] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[33] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[34] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[35] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[36] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[37] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[38] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[39] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[40] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[41] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[42] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[43] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[44] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[45] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[46] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[47] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[48] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[49] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[50] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[51] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[52] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[53] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[54] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[55] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[56] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[57] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[58] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[59] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[60] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[61] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[62] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[63] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[64] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[65] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[66] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[67] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[68] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[69] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[70] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[71] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[72] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[73] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[74] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[75] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[76] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[77] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[78] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[79] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[80] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[81] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[82] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[83] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[84] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[85] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[86] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[87] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[88] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[89] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[90] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[91] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[92] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[93] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[94] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[95] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[96] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[97] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[98] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[99] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[100] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[101] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[102] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[103] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[104] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[105] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[106] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[107] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[108] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[109] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[110] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[111] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[112] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[113] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[114] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[115] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[116] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[117] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[118] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[119] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[120] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[121] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[122] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[123] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[124] = *(float *)&hex_data;
hex_data = 0x1; rx_demod_data[125] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[126] = *(float *)&hex_data;
hex_data = 0x0; rx_demod_data[127] = *(float *)&hex_data;

}

#ifndef THREAD_PER_TASK
int  MOD_QPSK(int bitlen, unsigned char *bitstream, double *QPSK_real, double *QPSK_img, float *obuf) {  	
#else
void * MOD_QPSK(void *input) {  	

    int bitlen = ((struct args_qpsk*)input)->bitlen;
    unsigned char *bitstream = ((struct args_qpsk*)input)->bitstream;
    double *QPSK_real = ((struct args_qpsk*)input)->QPSK_real;
    double *QPSK_img = ((struct args_qpsk*)input)->QPSK_img;
    float *obuf = ((struct args_qpsk*)input)->obuf;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] TX-QPSK-MOD assigned to CPU: %d\n", sched_getcpu());
    #endif

    int i,j;
    char Dec_Num[bitlen/2];

    for (j = 0,i=0; j < bitlen; j=j+2)
    {
    
        if (bitstream[j] == 0 && bitstream[j + 1] == 0){ Dec_Num[i] = 0; }
        else if (bitstream[j] == 0 && bitstream[j + 1] == 1){ Dec_Num[i] = 1; }
        else if (bitstream[j] == 1 && bitstream[j + 1] == 0){ Dec_Num[i] = 2; }
        else if (bitstream[j] == 1 && bitstream[j + 1] == 1){ Dec_Num[i] = 3; }

        i++;
    }


for(i=0; i<bitlen/2; i++)
{ 

	if (Dec_Num[i] == 0){
    		QPSK_real[i]  = PM2;
    		QPSK_img[i] =  PM2;
    		}	
	else if (Dec_Num[i] == 1){
    		QPSK_real[i] = PM2;
    		QPSK_img[i]= PM1;
    		}
	else if (Dec_Num[i] == 2){
    		QPSK_real[i] = PM1; 
   		QPSK_img[i]= PM1;
    		}
	else if (Dec_Num[i] == 3){
     		QPSK_real[i]= PM1; 
     		QPSK_img[i]= PM2;
    		}
}
    for(i=0,j=0;i<bitlen;i=i+2)
    {
        obuf[i] = QPSK_real[j];
        obuf[i+1] = QPSK_img[j];
        j++;
    }

/*
    printf("\nQPSK Modulation output:\n");
    for(j=0;j<bitlen;j+=2)
    {
        printf("%+4.4f,%+4.4f\t", obuf[j], obuf[j+1]);
        if(j%8 == 6) printf("\n");
    }
    printf("\n");
*/

    return 0;
}

#ifndef THREAD_PER_TASK
int DeMOD_QPSK(int n, comp_t *ibuf, signed char *out) {
#else
void* DeMOD_QPSK(void *input) {

    int n = ((struct args_qpsk_demod *)input)->n;
    comp_t *ibuf = ((struct args_qpsk_demod *)input)->ibuf;
    signed char *out = ((struct args_qpsk_demod *)input)->out;
#endif

#ifdef ACC_RX_DEMOD
    memcpy(out, rx_demod_data, 128*sizeof(unsigned char));
    return 0;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] RX-QPSK-DEMOD assigned to CPU: %d\n", sched_getcpu());
    #endif

int i,j,m,z;
double d[4];
double DI[4];
double small;
char dec_bit[n];
int loc = 0;
double x2[n], y2[n];
//double r[n];
//int tens = 10, minusTens = -10;

DI[0] = sqrt(pow((-sqrt(2) / 2), 2) + pow((-sqrt(2) / 2), 2));
DI[1] = sqrt(pow((-sqrt(2) / 2), 2) + pow(sqrt(2) / 2,  2));
DI[2] = sqrt(pow(( sqrt(2) / 2), 2) + pow(( sqrt(2) / 2), 2));
DI[3] = sqrt(pow(sqrt(2) / 2 , 2) + pow((-sqrt(2) / 2), 2));

for(i=0,z=0; i<n;i++) {
    
    x2[i] = ibuf[z].real;
    y2[i] = ibuf[z].imag;
    
    z++;
    
d[0] = sqrt(pow(x2[i] - (-sqrt(2) / 2), 2) + pow(y2[i] - (-sqrt(2) / 2), 2));
d[1] = sqrt(pow(x2[i] - (-sqrt(2) / 2), 2) + pow(y2[i] -   sqrt(2) / 2,  2));
d[2] = sqrt(pow(x2[i] - ( sqrt(2) / 2), 2) + pow(y2[i] - ( sqrt(2) / 2), 2));
d[3] = sqrt(pow(x2[i] -   sqrt(2) / 2 , 2) + pow(y2[i] - (-sqrt(2) / 2), 2));

/*
for(i=0;i<4;i++)
{
printf("value of d[%d]:%f\n",i,d[i]);

}
printf("\n\n");
*/

small = d[0];
for (j = 0; j<4; j++){
    if (d[j]<=small){
        small = d[j];
        loc = j;
    }
}

//printf("i=%d,small:%f local:%d\n",i,small,loc);

//printf("%d,",loc);
if (loc == 0)
{ 
	dec_bit[i] = 0; 
#ifdef SOFTINPUT
        r[i]= (1-(small/DI[0]));
#endif
}
if (loc == 1)
{ 
	dec_bit[i] = 1; 
#ifdef SOFTINPUT
	r[i]= (1-(small/DI[1]));
#endif
}
if (loc == 2)
{ 
	dec_bit[i] = 2; 
#ifdef SOFTINPUT
	r[i]= (1-(small/DI[2]));
#endif
}

if (loc == 3)
{ 
	dec_bit[i] = 3; 
#ifdef SOFTINPUT
	r[i]= (1-(small/DI[3]));
#endif
}
    
}
/*
printf("\n");
for(i=0;i<n;i++)
{
printf("%d,",dec_bit[i]);
}
*/
//printf("\n\n");
for(m=0,j=0;m<=n*2;m=m+2) 
{
if (dec_bit[j] == 0)
    {
#ifdef SOFTINPUT
	out[m] = tens*r[j];
	out[m + 1] = tens*r[j];
#else
	out[m] = 0;
        out[m + 1] = 0 ;
#endif
    }

    else if (dec_bit[j] == 1)
    {
#ifdef SOFTINPUT
	out[m] = tens*r[j];
	out[m + 1] = minusTens*r[j];
#else
	out[m] = 0;
        out[m + 1] = 1;
#endif	
    }

    else if (dec_bit[j] == 2)
    {
#ifdef SOFTINPUT
out[m] = minusTens*r[j];
out[m + 1] = tens*r[j];
#else
	out[m] = 1;
        out[m + 1] = 0;
#endif
    }

    else if (dec_bit[j] == 3)
    {
#ifdef SOFTINPUT
	out[m] = minusTens*r[j];
	out[m + 1] = minusTens*r[j];
#else
 	out[m] = 1;
	out[m + 1] = 1;
#endif
    }
j++;
}

return 0;
}

