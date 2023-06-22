/*
(c) Copyright 2013 - 2016 Xilinx, Inc. All rights reserved.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising
under or in connection with these materials, including for any direct, or any
indirect, special, incidental, or consequential loss or damage (including loss
of data, profits, goodwill, or any type of loss or damage suffered as a result
of any action brought by a third party) even if such damage or loss was
reasonably foreseeable or Xilinx had been advised of the possibility of the
same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.
*/

//#include <stdio.h>
//#include <stdlib.h>
//#include "mmultadd.h"
#include "inverse.h"
/**
 *
 * Design principles to achieve II = 1
 * 1. Stream data into local RAM for inputs (multiple access required)
 * 2. Partition local RAMs into N/2 sub-arrays for fully parallel access (dual-port read)
 * 3. Pipeline the dot-product loop, to fully unroll it
 * 4. Separate multiply-accumulate in inner loop to force two FP operators
 *
 */
//A,Ai,B,Bi,C,Ci
void mmult (float A[N*M],float Ai[N*M], float B[M*N], float Bi[M*N] ,float C[N*N], float Ci[N*N])
{ int i ,j;

     float Abuf[N][M],Aibuf[N][M], Bbuf[M][N],Bibuf[M][N],Cbuf[N][N],Cibuf[N][N];
/*#pragma HLS array_partition variable=Abuf block factor=4 dim=2
#pragma HLS array_partition variable=Aibuf block factor=4 dim=2

#pragma HLS array_partition variable=Bbuf block factor=4 dim=1
#pragma HLS array_partition variable=Bibuf block factor=4 dim=1

#pragma HLS array_partition variable=Cbuf complete factor=4 dim=0
#pragma HLS array_partition variable=Cibuf complete factor=4 dim=0
*/

     for( i=0; i<N; i++) {
          for( j=0; j<M; j++) {
//#pragma HLS PIPELINE
               Abuf[i][j] = A[i*M + j];
               Aibuf[i][j] = Ai[i*M + j];

          }
     }

     for(i=0;i<M;i++)
     { for(j=0;j<N;j++)
		 {
//#pragma HLS PIPELINE
    	   Bbuf[i][j] = B[i * N + j];
    	   Bibuf[i][j] = Bi[i * N + j];

		 }
     }

     for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
/*#pragma HLS PIPELINE
#pragma HLS RESOURCE variable=Cbuf core=AddSub_DSP
#pragma HLS RESOURCE variable=Cibuf core=AddSub_DSP
  */             float result1 = 0,result2 = 0 , result3 = 0 , result4 =0 ;
               for (int k = 0; k < M; k++) {
/*#pragma HLS RESOURCE variable=result1 core=AddSub_DSP
#pragma HLS RESOURCE variable=result2 core=AddSub_DSP
#pragma HLS RESOURCE variable=result3 core=AddSub_DSP
#pragma HLS RESOURCE variable=result4 core=AddSub_DSP
*/
                    float term1 = Abuf[i][k] * Bbuf[k][j];
                    result1 += term1;
                    float term2 = Aibuf[i][k] * Bibuf[k][j];
                    result2 += term2;
                    float term3 = Abuf[i][k] * Bibuf[k][j];
                    result3 += term3;
                    float term4 = Aibuf[i][k] * Bbuf[k][j];
                    result4 += term4;
               }

               Cbuf[i][j] = result1 - result2;
               Cibuf[i][j] = result3 + result4;
               C[i*N+j]=Cbuf[i][j];
               Ci[i*N+j]=Cibuf[i][j];
          }
     }

}

