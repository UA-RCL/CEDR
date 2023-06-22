/*
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

#ifndef __TDSP_DEV_H
#define __TDSP_DEV_H

#include <linux/ioctl.h>
#include <tradio/ttypes.h>

//#define DSP_PRINTF_ENABLED

#define NUM_DSP_DEV     2
#define DSP0            0
#define DSP1            1

#ifndef _U32_
#define _U32_
    typedef unsigned int           	U32, *PU32;
    typedef volatile unsigned int	V32, *PV32;
#endif

typedef struct tDSPPRINF_ALLOC_INFO {
	unsigned int Version;
	unsigned int SizeFor2Ceva;		// The size of memory assigned to CEVA print area
	unsigned int pPhysCtxPtr;		// The physical address pointer to the CEVA print area 
} DSPPRINF_ALLOC_INFO, *PDSPPRINF_ALLOC_INFO;

/* 
 * Set the message of the device driver 
 */
#define TDSP_IOCTL_ALLOC_CRAM            _IOWR('d', 1, unsigned int)
#define TDSP_IOCTL_FREE_CRAM             _IOW('d', 2, unsigned int)
#define TDSP_IOCTL_BOOT                  _IOW('d', 3, PIMAGEDESC)
#define TDSP_IOCTL_WAIT_BOOT_COMP        _IO('d', 4)
#define TDSP_IOCTL_RUN                   _IOW('d', 5, PTCB)
#define TDSP_IOCTL_WAIT_RUN_COMP         _IOR('d', 6, PTCB)
#define TDSP_IOCTL_GEN_TM_TICK           _IOW('d', 7, unsigned int)
#define TDSP_IOCTL_GEN_SYMB_TM_TICK      _IOW('d', 8, unsigned int)
#define TDSP_IOCTL_ALLOC_DSPPRINTF_CTX  _IOWR('d', 9, PDSPPRINF_ALLOC_INFO)
#define TDSP_IOCTL_GET_DSPPRINTF_CTX    _IOWR('d', 10, PDSPPRINF_ALLOC_INFO)
#define TDSP_IOCTL_CTRL                 _IOWR('d', 11, unsigned int)

#define TDSP_IOCTL_CTRL_MASK__DISABLED        (0)
#define TDSP_IOCTL_CTRL_MASK__PACKED_TCB (1 << 0)

#define HAL_IRQ_TIMER                    (32+112)

/* 
 * The name of the device file 
 */
#define TDSP_DEVICE_NAME "tdsp"

#define CEVACTRL_NO_READ(mapped_base, regaddr) (mapped_base + ((regaddr) - CEVA_RESET) / 4)
#define CEVACTRL(mapped_base, regaddr)  (*CEVACTRL_NO_READ(mapped_base, regaddr))

#define DSP_BOOT_DESC_ADDR      (TRANSCEDE_CRAM_BASE + 0x800)
#define DSP_EXT_SEGMENTS        (TRANSCEDE_CRAM_BASE + 0x1000)
#define DSP_INT_SEGMENTS        (TRANSCEDE_CRAM_BASE + 0x2000)

#define INTCONF_LEVEL           (UINT32)1
#define INTCONF_EDGE            (UINT32)3

#define A9IC_DISTR_BASEADDR     0xFFF01000

//typedef unsigned int UINT32, *PUINT32;
//typedef volatile unsigned int VUINT32, *PVUINT32;

typedef struct tagIMAGEDESC {
    int boot_vect;
    int size;
    void *data;
} IMAGEDESC, *PIMAGEDESC;

typedef struct tagDSPBOOTMEM {
    unsigned int Dsp0Tcb; // 0x00
    unsigned int Dsp0BootInfo[3];
    unsigned int Dsp1Tcb; // 0x10
    unsigned int Dsp1BootInfo[3];
    unsigned int res0[0x1F8];
    unsigned int DspEntryPoint;
} DSPBOOTMEM, *PDSPBOOTMEM;

typedef struct tDSPBOOTDESC {
    UINT32  next;	/* next pointer in external memory */
    UINT32  code;	/* use program or data memory DMA */
    UINT32  length;	/* section length in bytes */
    UINT32  internal;	/* internal load address for DMA */
    UINT32  external;	/* external load address for DMA */
    UINT32  pad[3];
} DSPBOOTDESC, *PDSPBOOTDESC;

typedef struct _A9ICDISTR_
{
    VUINT32 Ctrl;          /**< Distributor Control Register (ICDDCR)      */
    VUINT32 Type;          /**< Interrupt Controller Type Register (ICDICTR)    */
    VUINT32 DistID;        /**< Distributor Implementer Identification Register (ICDIIDR) */
    VUINT32 Res[29];       /**< Reserved             */
    VUINT32 Security[32];  /**< Interrupt Security Registers (ICDISRn)      */
    VUINT32 SetEnable[32]; /**< Interrupt Set-Enable Registers (ICDISERn)     */
    VUINT32 ClrEnable[32]; /**< Interrupt Clear-Enable Registers (ICDICERn)    */
    VUINT32 SetPend[32];   /**< Interrupt Set-Pending Registers (ICDISPRn)     */
    VUINT32 ClrPend[32];   /**< Interrupt Clear-Pending Registers (ICDICPRn)    */
    VUINT32 ActBit[32];    /**< Active Bit Registers (ICDABRn)        */
    VUINT32 Res3[32];      /**< Reserved             */
    VUINT32 Priority[255]; /**< Interrupt Priority Registers (ICDIPRn)      */
    VUINT32 Res4;          /**< Reserved             */
    VUINT32 Target[255];   /**< Interrupt Processor Targets Registers (ICDIPTRn)   */
    VUINT32 Res5;          /**< Reserved             */
    VUINT32 Config[64];    /**< Interrupt Configuration Registers (ICDICFRn)    */
    VUINT32 PPIStatus;     /**< PPI Status Register */
    VUINT32 SPIStatus[31]; /**< SPI Status Register */
    VUINT32 Res6[32];      /**< IMPLEMENTATION DEFINED registers       */
    VUINT32 Res7[64];      /**< Reserved             */
    VUINT32 SoftGenInt;    /**< Software Generated Interrupt Register (ICDSGIR)   */
    VUINT32 Res8[51];      /**< Reserved             */
    VUINT32 ID[12];        /**< Identification registers         */

} A9ICDISTR, *PA9ICDISTR;

typedef struct _st_stack_all_regs{
    UINT32 l3_bknest1;
    UINT32 l3_bknest0;
    UINT32 l2_bknest0;
    UINT32 l2_bknest1;
    UINT32 l1_modu2;
    UINT32 l1_modu3;
    UINT32 g4;
    UINT32 g5;
    UINT32 g6;
    UINT32 g7;
    UINT32 modv1;
    UINT32 modv0;
    UINT32 vpr0[8];
    UINT32 vc0[8];
    UINT32 vob0[8];
    UINT32 vob0e[8];
    UINT32 voa0[8];
    UINT32 voa0e[8];
    UINT32 vih0[8];
    UINT32 vig0[8];
    UINT32 vif0[8];
    UINT32 vie0[8];
    UINT32 vid0[8];
    UINT32 vic0[8];
    UINT32 vib0[8];
    UINT32 via0[8];
    UINT32 mod0;
    UINT32 mod1;
    UINT32 mod2;
    UINT32 modg;
    UINT32 retreg;
    UINT32 retregb;
    UINT32 retregi;
    UINT32 retregn;
    UINT32 modp;
    UINT32 modq;
    UINT32 a4_7[4];
    UINT32 a20_23[4];
    UINT8 ae4_7[4];
    UINT8 ae20_23[4];
    UINT32 l1_bknest0;
    UINT32 l1_bknest1;
    UINT32 s2;
    UINT32 s3;
    UINT32 a12_19[8];
    UINT8 ae12_19[8];
    UINT32 s0;
    UINT32 s1;
    UINT32 r4_7[4];
    UINT32 a8_11[4];
    UINT32 a0_3[4];
    UINT8 ae8_11[4];
    UINT8 ae0_3[4];
    UINT32 modu0;
    UINT32 modu1;
    UINT32 l0_bknest0;
    UINT32 l0_bknest1;
    UINT32 l0_modu2;
    UINT32 l0_modu3;
    UINT32 g4_7[4];
    UINT32 g0_3[4];
    UINT32 r0_3[4];
}st_stack_all_regs;

#endif
