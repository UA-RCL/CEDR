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

#ifndef _TCB_H_
#define _TCB_H_

#define MAX_IOBUF_DESC      16

typedef struct tIOBufDesc
{
    void *IOBufPtr;
    unsigned int IOBufControl;
} TIOBufDesc, *PTIOBufDesc;


#define CEVA_VERSION_X_MASK	(0x0F << 28)
#define CEVA_VERSION_Y_MASK	(0x0F << 24)
#define CEVA_VERSION_Z_MASK	(0xFF << 16)
#define CEVA_VERSION_RC_MASK	(0xFF << 8)
#define CEVA_STATUS_MASK	(0xFF << 0)
#define CEVA_VERSION_MASK	(CEVA_VERSION_X_MASK | CEVA_VERSION_Y_MASK | CEVA_VERSION_Z_MASK | CEVA_VERSION_RC_MASK)

#define CEVA_VERSION_X(x)	((x & CEVA_VERSION_X_MASK) >> 28)
#define CEVA_VERSION_Y(x)	((x & CEVA_VERSION_Y_MASK) >> 24)
#define CEVA_VERSION_Z(x)	((x & CEVA_VERSION_Z_MASK) >> 16
#define CEVA_VERSION_RC(x)	((x & CEVA_VERSION_RC_MASK) >> 16)
#define CEVA_VERSION(x)		((x & CEVA_VERSION_MASK) >> 8)
#define CEVA_STATUS(x)		((x & CEVA_STATUS_MASK) >> 0)

// if changing, please make sure to update same defines here:
// package/mspd/shmsemadrv/src/shmsemadrv.c
#define API_EVENT_MASK			(0x0000000F)
#define CEVA_TO_ARM_API_EVENT		(0x0000FFF0)
#define CEVA_TO_ARM_IRQ_EVENT		(0x0000AAA0)

#define CEVA_TO_ARM_API_EVENT_MASK	(~API_EVENT_MASK)
#define IS_CEVA_TO_ARM_API_EVENT(x)	(((x) & CEVA_TO_ARM_API_EVENT_MASK) == CEVA_TO_ARM_API_EVENT)
#define IS_CEVA_TO_ARM_IRQ_EVENT(x)	((x) == CEVA_TO_ARM_IRQ_EVENT)

#define EVM_TCB_EVENT_TYPE_NONE		(0x00000000)
#define EVM_TCB_EVENT_TYPE_TCB		(0x00000001)
#define EVM_TCB_EVENT_TYPE_EVENT_CEVA_DL (0x0000FFF0) //CEVA_EVENT_DL
#define EVM_TCB_EVENT_TYPE_EVENT_CEVA_UL (0x0000FFF1) //CEVA_EVENT_UL

#define EVM_TCB_SIZE            (128)
#define CRAM_ARM_MSG_ADDR(id) ((TCB *)(TRANSCEDE_CRAM_BASE + EVM_TCB_SIZE * id))

typedef struct tTCB {
              union {
/* 0x00 */        struct tTCB     *NextTcb;
/* 0x00 */        unsigned int    Event;
              };
/* 0x04 */    unsigned int    TaskID;
/* 0x08 */    unsigned int    ResourceID;
              union {
/* 0x0C */        unsigned int FVS;
                  struct {
/* 0x09 */            char    Status;
/* 0x0A */            char    FirmwareVersionRC;
/* 0x0B */            char    FirmwareVersionZ;
                      char    FirmwareVersionY : 4;
/* 0x0C */            char    FirmwareVersionX : 4;
                  };
                  struct {
/* 0x09 */            unsigned int : 8;	// Status
/* 0x0A */            unsigned int FirmwareVersion : 24;
                  };
              };
/* 0x10 */    void (* cbDone)(struct tTCB *Tcb);
/* 0x14 */    void            *ContextPtr;
/* 0x18 */    unsigned int    ContextLen;
/* 0x1C */    void            *IOControlPtr;
/* 0x20 */    unsigned int    IOControlLen;
/* 0x24 */    void            *InputDataPtr;
/* 0x28 */    unsigned int    InputDataLen;
/* 0x2C */    void            *OutputDataPtr;
/* 0x30 */    unsigned int    OutputDataLen;
/* 0x34 */    unsigned int    ExecTicks;
/* 0x38 */    unsigned int    Res0[2];
} TCB, *PTCB;

#define IN_BUF              0x40000000
#define OUT_BUF             0x80000000
#define INOUT_BUF           0xC0000000
#define BUF_SIZE_MASK       0x00FFFFFF
#define IOCONTR_MASK        0xFF000000
#define IO_MASK             0xc0000000

#define MLOG_SUBTASK_SIZE  50 // in U32 words

////////////////////////////////////////////
//define TCB Status types
////////////////////////////////////////////
#define STATUS_READY        0  //tcb is not scheduleed yet
#define STATUS_COMPLETE     1  //tcb has launched and has finished execution
#define STATUS_RUNNING      2  //tcb has been launched but NOT done yet
#define STATUS_INV_TASKID   3 //invalid task ID
#define STATUS_PARAM_ERROR  4 //error in parameters

////////////////////////////////////////////
//define TCB pri
////////////////////////////////////////////
#define TCB_DEFAULT_PRI    0x10000000   //pri goes from 0 the highest pri
#define MAX_NUM_TCB        512         //4096
#define MAX_NUM_RSRC       16

////////////////////////////////////////////////////////////////////////////////////////
// All Task IDs defined across PHY code
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////
// 1. LTE BS Tx (DL) Tasks
////////////////////////////////////////////
#define TASKID_CRC24A                            100
#define TASKID_FEC_TX                            101
#define TASKID_SCRAMBLER                         102
#define TASKID_PDCCH_SCRAMBLER                   103
#define TASKID_LOAD_SETUPSYMBDATAINPBUFS         104
#define TASKID_PSUEDORANDOM_SEQGEN               105
#define TASKID_LTE_DLPILOT                       106
#define TASKID_LTE_BSTXSYNCCH                    107
#define TASKID_LTE_DLCONTROL                     108
#define TASKID_LTE_BSTXPHICH                     109
#define TASKID_LTE_SBITLVPDCCH                   110
#define TASKID_LTE_MULTICHAN_MODULATION          111
#define TASKID_LTE_MAPPER_MULTICHAN              112
#define TASKID_LTE_MAPPER                        113
#define TASKID_LTE_LAYERMAPPER_MULTICHAN         114
#define TASKID_LTE_LAYERMAP                      115
#define TASKID_LTE_PRECODER_MULTICHAN            116
#define TASKID_LTE_PRECODER                      117
#define TASKID_LTE_DLRESELEMMAPPER_MULTICHAN     118
#define TASKID_LTE_RCRELMMAP                     119
#define TASKID_LTE_DLREMAPPING                   120
#define TASKID_LTE_SETUPIFFTBUFS                 121
#define TASKID_IFFT                              122
#define TASKID_PAPR_RCF                          123
#define TASKID_ADDCP                             124
#define TASKID_LTE_MULTICHAN_MODULATION_SETUP    125

////////////////////////////////////////////
// 2. LTE BS Rx (UL) Tasks
////////////////////////////////////////////
#define TASKID_RX_GEN_HCS                        150
#define TASKID_RX_REMOVE_HCS                     151
#define TASKID_FFT                               152
#define TASKID_RX_SWAP_FFTOUT                    153
#define TASKID_RX_GEN_ULPILOT                    154
#define TASKID_RX_ULPILOT                        155
#define TASKID_RX_ULPILOT_PUCCH                  156
#define TASKID_RX_FDMAPHYSCHAN                   157
#define TASKID_RX_CHANEST_P1                     158
#define TASKID_RX_CHANEST_P2                     159
#define TASKID_RX_CHANEST_P3                     160
#define TASKID_RX_CHANEST_P4                     161
#define TASKID_RX_CHANEST_PUCCH                  162
#define TASKID_LTE_MULTICHAN_DEMODULATION        163
#define TASKID_RX_EXP_EQ16                       164
#define TASKID_LTE_MRCOM                         165
#define TASKID_RX_FEQ                            166
#define TASKID_NORM32TO16                        167
#define TASKID_IDFT                              168
#define TASKID_BLOCK_DENORM                      169
#define TASKID_CAZACAVG                          170
#define TASKID_LTE_DEMAPPER                      171
#define TASKID_ROTATE                            172
#define TASKID_DESCRAMBLER                       173
#define TASKID_GETDETHARD                        174
#define TASKID_UPDATEBITS                        175
#define TASKID_FEC_RX_CODING                     176
#define TASKID_FEC_PUCCH                         177
#define TASKID_FEC_RX                            178

////////////////////////////////////////////
// Service Tasks
////////////////////////////////////////////
#define TASKID_DMA_DRAM                         201
#define TASKID_MEM_COPY                         202
#define TASKID_MEM_SET                          203
#define TASKID_HEAP_ALLOC                       204
#define TASKID_HEAP_FREE                        205
#define TASKID_DELAY                            206
#define TASKID_IDM_TEST                         207
#define TASKID_PRINTF_ENA                       208
#define TASKID_DMA_QUEUE_DRAM                   209
#define TASKID_MCCI_WRITE                       210
#define TASKID_MCCI_READ                        211
#define TASKID_IRQ_TRAINING                     212
#define TASKID_PCACHE_TEST                      213
#define TASKID_TM_TICK                          214
////////////////////////////////////////////
// 4. Mobile Station Tasks
////////////////////////////////////////////
#define TASKID_MSRX_FEQ                          300
#define TASKID_MSRX_FINDMAX                      301
#define TASKID_MSRX_EXP_EQ32                     302
#define TASKID_MSRX_EXP_EQ16                     303
#define TASKID_LTE_MS_CHAN_EST                   304
#define TASKID_RX_FINDMAX                        305
#define TASKID_RX_EXP_EQ32                       306

////////////////////////////////////////////
// ASU Tasks
////////////////////////////////////////////
#define TASKID_ASU_TEST				 310
#define TASKID_ASU_TEST2			 311
#define TASKID_ASU_ADD				 312
#define TASKID_ASU_FFT				 313
#define TASKID_ASU_FFT32			 314
#define TASKID_ASU_IFFT32			 315
#define TASKID_ASU_FFT64			 316
#define TASKID_ASU_IFFT64			 317
#define TASKID_ASU_VITERBIK7                     318
#define TASKID_ASU_END				 320
////////////////////////////////////////////
// 5. Un-used Tasks (Legacy)
////////////////////////////////////////////
#define TASKID_RX_CHANESTPUCCH                   400
#define TASKID_SC_RNDMZ                          401
#define TASKID_PREMAPPER                         402
#define TASKID_PERM_TX                           403
#define TASKID_SC_DERNDMZ                        404
#define TASKID_LOADIFFT                          405
#define TASKID_SENDTXOUT                         406
#define TASKID_POSTFFT                           407
#define TASKID_CHEST                             408
#define TASKID_PILOTS_RX                         409
#define TASKID_SUBCHROTATION_RX                  410
#define TASKID_FFT_NORM_RX                       411
#define TASKID_STC_ENCODER                       412
#define TASKID_MACRXIND                          413
#define TASKID_MACRXRANGINGIND                   414
#define TASKID_STORE_RANGING                     415
#define TASKID_INIT_RANGING                      416
#define TASKID_ALLOC_RANGING                     417
#define TASKID_RANGING_CORR                      418
#define TASKID_RANGING_SWAP                      419
#define TASKID_RANGING_PROC                      420
#define TASKID_RANGING_DET                       421
#define TASKID_IFFT_RX                           422
#define TASKID_FFDEMAPPER                        423
#define TASKID_MACRXFFIND                        424
#define TASKID_ARMTEST                           425
#define TASKID_DEMAPPER                          426
#define TASKID_MRCOM                             427
#define TASKID_LTE_FFT                           428
#define TASKID_LTE_FEC                           429
#define TASKID_MAPPER                            430
#define TASKID_LTE_LOADIFFT                      431
#define TASKID_LTE_TXOUT                         432
#define TASKID_STC                               433
#define TASKID_SUBBLOCKINTLVR_PDCCH              434
#define TASKID_LTE_MAPPER_SYMB                   435
#define TASKID_DLRESELMAPPER                     436
#define TASKID_DLRSRCELEMASIGN                   437
#define TASKID_RX_RGCONSTELLATION_PUCCH          438
#define TASKID_FECMEMALLOC                       439
#define TASKID_CRC24B                            440
#define TASKID_LOAD_SYMBOL                       441
#define TASKID_RX_DEMUXOFDMSYM                   442
#define TASKID_RX_RGCONSTELLATION                443
#define TASKID_LTE_IFFT                          444
#define TASKID_LTE_INITTX                        445



#endif
