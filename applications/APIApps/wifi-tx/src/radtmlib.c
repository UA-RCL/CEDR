/**
 * @file radtimers.c
 *
 * @brief T2200/T3300 Radio Timer library source code
 *
 * Copyright(c) 2007-2014 Intel Corporation. All rights reserved.
 *
 *   This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
 * General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
 * The full GNU General Public License is included in this distribution 
 * in the file called LICENSE.GPL.
 *
 * Contact Information:
 * Intel Corporation
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <errno.h>

#include "p2vconv.h"
#include "radtimers.h"

#define SYS_REF_CLK     25
#define AXI_CLK_DIV_CNTRL               0xF4CF004C  /**< AXI Clock Divider Control Register */

// Static global variables
static volatile PRADTMREGS     pRadTmRegs;
static unsigned int            SysBusClockFreq;
static volatile unsigned int * pTimerReg;

// Function prototypes:
unsigned int getChipCount (volatile PFSYNCTMREGS pFSyncTm);
unsigned int getFrameCount(volatile PFSYNCTMREGS pFSyncTm);

unsigned int SysTimerGetTick(void)
{
    return *pTimerReg;
}

void RadTmSetReg(volatile unsigned int *pReg, volatile unsigned int Value)
{
    volatile unsigned int start_tick, tick;
    volatile unsigned int timeout;
    
    *pReg = Value;

    timeout    = 2*SysBusClockFreq;
    start_tick = SysTimerGetTick();
    do
    {
        tick = SysTimerGetTick();
    }
    while ((tick - start_tick) < timeout);
}

int RadioTimerOpen(void)
{
    volatile PFSYNCTMREGS   pFSyncTm;
    volatile unsigned int * pPllRegs;
    volatile unsigned int * pAxiDivCntrl;
    unsigned int            PllLsb, PllMsb, ClkDiv;
    
    p2vconv_init();
    
    pPllRegs        = p2vconv(PLL_M_LSB(1), 8);
	if (!pPllRegs)
		return 0;
    PllLsb          = *pPllRegs++;
    PllMsb          = *pPllRegs++;
    pAxiDivCntrl    = p2vconv(AXI_CLK_DIV_CNTRL, 4);
	if (!pAxiDivCntrl)
		return 0;
    ClkDiv          = *pAxiDivCntrl;
    SysBusClockFreq = SYS_REF_CLK * ((PllMsb << 8) | PllLsb) / 5 / ClkDiv;

    pTimerReg = p2vconv(0xFE050004, 4);
    if (!pTimerReg)
        return 0;
    
    pRadTmRegs = p2vconv(RAD_TM_BASEADDR, sizeof(RADTMREGS));

    if (pRadTmRegs) 
    {
        pFSyncTm = &pRadTmRegs->FSyncTimers[0];
        //printf("SysBusClockFreq=%d PRESCALE_CLK_CNTRL=%x\n", SysBusClockFreq, pFSyncTm->PRESCALE_CLK_CNTRL);
    }

    return 1;
}

void RadioTimerClose(void)
{
    p2vconv_cleanup();
}

void RadioTimerQuery(unsigned int *ChipCounter, unsigned int *FrameCounter)
{
    volatile PFSYNCTMREGS pFSyncTm = &pRadTmRegs->FSyncTimers[0];

    // Take a first look at the frame count and chip count
    *FrameCounter = getFrameCount(pFSyncTm);
    *ChipCounter  = getChipCount(pFSyncTm);
    
    // Re-read the frame counter to check we definitely haven't bridged a chip counter wrap
    unsigned int checkFrameCounter = getFrameCount(pFSyncTm);
    if (checkFrameCounter != *FrameCounter)
    {
        // May well have bridged a wrap, so re-read chip counter
        *ChipCounter  = getChipCount(pFSyncTm);
        *FrameCounter = checkFrameCounter;
    }
}

// Provide an approximation to an atomic read of the chip count
unsigned int getChipCount(volatile PFSYNCTMREGS pFSyncTm)
{
    // Read the slot counter first - magic numbers taken from Alex's original code
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (3<<10) | 3);
    unsigned int slotCntr = pFSyncTm->COUNTER_CURR_VAL;

    // Read the chip counter
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (2<<10) | 3);
    unsigned int chipCntr = pFSyncTm->COUNTER_CURR_VAL;

    // Re-read the slot counter.  If it has changed then likelihood is we have
    // bridged a counter wrap and need to reread the chip counter.
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (3<<10) | 3);
    if (pFSyncTm->COUNTER_CURR_VAL != slotCntr)
    {
        slotCntr = pFSyncTm->COUNTER_CURR_VAL;
        RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (2<<10) | 3);
        chipCntr = pFSyncTm->COUNTER_CURR_VAL;
    }

    return ((CHIP_SRATE * slotCntr) + chipCntr);
}


// Provide an approximation to an atomic read of the frame count
unsigned int getFrameCount(volatile PFSYNCTMREGS pFSyncTm)
{
    // Read most significant portion first - magic numbers from Alex's original code
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (5<<10) | 3);
    unsigned int msp = pFSyncTm->COUNTER_CURR_VAL;

    // Read least significant portion
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (4<<10) | 3);
    unsigned int lsp = pFSyncTm->COUNTER_CURR_VAL;

    // re-read most significant portion. If it has changed then need to reread the least
    // significant portion as we have bridged a gap.
    RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (5<<10) | 3);
    if (pFSyncTm->COUNTER_CURR_VAL != msp)
    {
        msp = pFSyncTm->COUNTER_CURR_VAL;
        RadTmSetReg(&pFSyncTm->TIMER_CNTRL, (1<<13) | (4<<10) | 3);
        lsp = pFSyncTm->COUNTER_CURR_VAL;
    }

    return ((msp << 13) | lsp);
}
