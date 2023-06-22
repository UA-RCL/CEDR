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
// A,Ai,B,Bi,C,Ci
void mmult4(double A[N * N], double Ai[N * N], double B[N * N], double Bi[N * N], double C[N * N], double Ci[N * N]) {
	int i, j;

	double Abuf[N][N], Aibuf[N][N], Bbuf[N][N], Bibuf[N][N];
#pragma HLS array_partition variable = Abuf block factor = 4 dim = 2
#pragma HLS array_partition variable = Aibuf block factor = 4 dim = 2

#pragma HLS array_partition variable = Bbuf block factor = 4 dim = 1
#pragma HLS array_partition variable = Bibuf block factor = 4 dim = 1

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
#pragma HLS PIPELINE
			Abuf[i][j] = A[i * N + j];
			Aibuf[i][j] = Ai[i * N + j];
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
#pragma HLS PIPELINE
			Bbuf[i][j] = B[i * N + j];
			Bibuf[i][j] = Bi[i * N + j];
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
#pragma HLS PIPELINE
			double result1 = 0, result2 = 0, result3 = 0, result4 = 0;
			for (int k = 0; k < N; k++) {
				double term1 = Abuf[i][k] * Bbuf[k][j];
				result1 += term1;
				double term2 = Aibuf[i][k] * Bibuf[k][j];
				result2 += term2;
				double term3 = Abuf[i][k] * Bibuf[k][j];
				result3 += term3;
				double term4 = Aibuf[i][k] * Bbuf[k][j];
				result4 += term4;
			}
			C[i * N + j] = result1 - result2;
			Ci[i * N + j] = result3 + result4;
		}
	}
}
