/*
 * Copyright(c) 2007-2014 Intel Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
 * The full GNU General Public License is included in this distribution
 * in the file called LICENSE.GPL.
 *
 * Contact Information:
 * Intel Corporation
 */
#ifndef __DRADIO_H__
#define __DRADIO_H__

/** @brief structure for the CPRI initialization */
typedef struct tagCPRIPARAM {
	unsigned int Rate;
	unsigned int SamlingRate;
	unsigned int Width;
	unsigned int Loopback;
	unsigned int Res[4];
} CPRIPARAM, *PCPRIPARAM;

/// Ducatti channels mask/id
#define DUCATTI_DUC_A_MSK 1
#define DUCATTI_DUC_B_MSK 2
#define DUCATTI_DDC_A_MSK 4
#define DUCATTI_DDC_B_MSK 8
#define DUCATTI_DDC_AUX_MSK 16

/// Ducatti loopbacks
#define DUCATTI_NO_LB 0          // no loopback
#define DUCATTI_CPRI_E2E_LB 1    // SERDES TX-to-RX loopback, no looptiming, CPRI Slave in master mode
#define DUCATTI_CPRI_FRAME_LB 2  // CPRI RX-to-TX full frame loopback

/// Ducatti supported CPRI rates
#define DCPRI_RATE_614 1    // 0.614g (CPRI)
#define DCPRI_RATE_1228 2   // 1.228g (CPRI)
#define DCPRI_RATE_2456 4   // 2.456g (CPRI)
#define DCPRI_RATE_3070 5   // 3.070g (CPRI)
#define DCPRI_RATE_4912 8   // 4.912g (CPRI)
#define DCPRI_RATE_6140 10  // 6.14g (CPRI)

int DucattiCpriInit(unsigned int id, PCPRIPARAM pCpriParam);
int DucattiChRst(unsigned int id, unsigned int mask);

#endif
