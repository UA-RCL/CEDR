#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#define debug 1

int Ch_inter_Eqau(int N, int PI, float *RxData, float *Spilot, float *Ypilot,float *EqRxData) 
{

int i,j; 
int P = N/PI;
float DataResp[150];
float PilotResp[PI*2];
float Slope1,Slope2,Inter1,Inter2;

//intilization 

for(i=0;i<150;i++)
{
	DataResp[i]=0;	 
}
for(i=0;i<PI*2;i++)
{

	PilotResp[i]=0;
}

//............Channel Estimation

for(i=0;i<PI*2;i++)
{
	PilotResp[i] = Ypilot[i] / Spilot[i];
}

#ifdef debug 
printf("\nChannelEstimation\n");
for(i=0;i<PI*2;i++)
{
	printf("%f,",PilotResp[i]);
}
printf("\n\n");
#endif

for(i=0,j=0;i<=(PI*2);i++)
{
	DataResp[(i*P*2)]= PilotResp[j];
	DataResp[(i*P*2)+1]= PilotResp[j+1];
        j=j+2;	

}

#ifdef debug
printf("DataResp:\n");
for(i=0;i<N*2;i++)
{
	printf("%f,",DataResp[i]);

}
printf("\n\n");
#endif


for(i=0;i<(PI*2)-2;i=i+2)
{
	Slope1 = (PilotResp[i+2] - PilotResp[i]) / PI;
    	Slope2 = (PilotResp[i+3] - PilotResp[i+1]) / PI;
    	Inter1 = PilotResp[i];
    	Inter2 = PilotResp[i+1];
	
	for(j=1;j<(PI*4*2);j=j+2)
	{
        	DataResp[(i*PI*PI)+(j-1)] = Slope1 * ((j-1)/8) + Inter1;
      		DataResp[(i*PI*PI)+(j)] = Slope2 * ((j-1)/8) + Inter2;
   	}

}

#ifdef debug
printf("After Interpolation\n");
for(i=0;i<N*2;i++)
{
	printf("%f,",DataResp[i]);
}
printf("\n\n");
#endif

//Slope1 = (PilotResp[i+3] - PilotResp[i+1]) / PI;
//Slope2 = (PilotResp[i+4] - PilotResp[i+2]) / PI;
//Slope =Slope1+Slope2;
i=(PI-1)*2;
Inter1 = PilotResp[i];
Inter2 = PilotResp[i+1];

//k=96 ->128 for upsizing 

for(j=1;j<=(PI*4*2)-1;j=j+2)
{

   DataResp[(i*16)+(j-1)] = Slope1 * ((j-1)/2) + Inter1;
   DataResp[(i*16)+(j)] = Slope2 * ((j-1)/2) + Inter2;

}

#ifdef debug
printf("Data After upsizing\n");
for(i=0;i<N*2;i++)
{
	printf("%f,",DataResp[i]);
}
printf("\n\n");
#endif 


for(i=0;i<(N*2);i++)
{
	EqRxData[i] = RxData[i] / DataResp[i];	
}

#ifdef debug
printf("Channel Qualization\n");
for(i=0;i<(N*2);i++)
{
	printf("%f,",EqRxData[i]);
}
printf("\n\n");
#endif

return 0;
}

