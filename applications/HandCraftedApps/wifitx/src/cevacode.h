/** @file svsrhwdiag.h
 *
 * @brief Hardware/software diagnostics controlled by the supervisor
 * @author Intel Corporation
 *
 *
 * Copyright 2009-2014 Intel Corporation All Rights Reserved.
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
 *
 * COPYRIGHT&copy; 2009-2014 Intel Corporation.
 */

#include "tdsp.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <fcntl.h>


#define HW_MEM_DEV                     "/dev/mem"
#define HW_CEVA_DEV                    "/dev/tdsp0"
#define HW_CEVA_IMAGE_BOOT             "/usr/sbin/cevaboot"
#define PROFILE_PATH_IMAGE             "/tftpboot/rfprofile.bin"


#define HW_DIAG_TX_BUF_ADDR            0x0
#define HW_DIAG_RX_BUF_ADDR            0x0

#define HW_DIAG_LB_MODE_OFF            0
#define HW_DIAG_LB_MODE_RX_TX          1
#define HW_DIAG_LB_MODE_TX_RX          2

#define HW_DIAG_RC_OPEN_DEV_ERROR      1
#define HW_DIAG_RC_MAP_BUF_ERROR       2
#define HW_DIAG_RC_SPI_INIT_ERROR      3
#define HW_DIAG_RC_ALLOC_ERROR         4
#define HW_DIAG_RC_NOT_INITED          5
#define HW_DIAG_RC_RADION_INIT_ERROR   6
#define HW_DIAG_RC_UNSUP_API           7
#define HW_DIAG_RC_CEVA_INIT_ERR       8
#define HW_DIAG_RC_INIT_DEV_ERROR      9

#define HW_DIAG_RADMODE_PULSE          0    // Pulse Radio Mode
#define HW_DIAG_RADMODE_LEVEL          1    // Level Radio Mode
#define HW_DIAG_RADMODE_TDD_PULSE      2    // TDD Pulse Radio Mode

#define TRAD_IQ_LOG_BUF_SIZE                (30720000)

#define NUM_DSP_DEV     2
#define DSP0            0
#define DSP1            1
#define TRANSCEDE_CRAM_BASE 0xF3000000
#define TCB_OFFSET  0x200

#define SZF_CMPX    2

int LoadDsp();


typedef struct _HW_DIAG_CTX_
{
    int             CevaDevHandle;
    U32*            TxBufPtr;
    U32*            RxBufPtr;
    volatile U32 *  pCevaCtrlRegs;
    U32 *           cramptr; // CRAM address 
    char *          CevaImagePath;
} cevadata;
