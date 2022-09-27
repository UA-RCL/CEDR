#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "baseband_lib.h"
#include "qpsk_Mod_Demod.h"
#define PM1 sqrt(2)/2
#define PM2 -sqrt(2)/2
//#define SOFTINPUT 1

int  MOD_QPSK(int bitlen, unsigned char *bitstream, double *QPSK_real, double *QPSK_img, float *obuf)
{  	
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

int DeMOD_QPSK(int n, comp_t *ibuf, signed char *out)
{
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

