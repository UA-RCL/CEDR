/**
 * @file test.c
 *
 * @brief Main entry and command parsing software
 * Linux Analog Devices AD936X radio chipset test code
 *
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

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/wait.h>
#include "tradio.h"

#include "ad9361radio.h"
#include "spi.h"
#include "trioctl.h"
#include "dradio.h"
#include "rf_interface.h"
#include <libradio/radio.h>

//#define RADTM_TEST_ENABLE

extern int  RadioTimerOpen(void);
extern void RadioTimerClose(void);
extern void RadioTimerQuery(unsigned int *ChipCounter, unsigned int *FrameCounter);

// 1 MHz sine waveform
extern unsigned int sin1mhziqnum;
extern signed short sin1mhziqbuf[];
extern unsigned int ltedlqam16iqnum;
extern signed short ltedlqam16iqbuf[];

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

#define FRAMELENGTH 804 

signed short txdata[15360*4*100];
signed short rxdata[15360*4*100];
RADCONF RadConfig;


/*! @brief Array with Transcede radio interface driver Linux device names */
static const char *rad_device[2] =  {
    "/dev/tradio0",
    "/dev/tradio1"
};

static int RadId                = -1;
static int radio_init           = 0;
static int radio_mode           = -1;
static int radio_loopback       = 0;
static int radio_test_time      = 0;
static int radio_playback_sid   = 0;
static int gpio_rf_sync         = 0;
static int radio_band           = -1;
static int radio_sampling_rate  = -1;
static int radio_reset_request  = 0;
static int radio_sync_start     = 0;
static int ducatti_cpri_rate    = 0;    // CPRI interface disabled by default
static int radio_auxdac_width   = 0;
static int radio_auxdac_max     = 0;
static int radio_auxdac_value   = 0;

static int radio_change_requested = 0;

static int       use_libradio   = 0;
static RadioMode libradio_mode  = RADIO_MODE_NULL;
static RadioConf libradio_conf  = RADIO_CONF_NULL;
static RadioDebug radio_debug   = RADIO_DEBUG_WARN;

static char* script_fn          = 0;
static char* capture_fn         = 0;
static char* sample_fn          = 0;

static char* script_init_fn     = 0;

static unsigned int pattern_buffer[2*15360];

static unsigned int rad_timer_records[1000];

static void radio_reset(void);

static int RadDrvCheckResult(unsigned int Antenna,
                             unsigned int i,
                             unsigned int mask,
                             unsigned int pRxDataPhys,
                             unsigned int BufSize,
                             unsigned int pRefBufPhys,
                             unsigned int RefBufSize
                            );

static int SaveCapturedSignal(unsigned int pa, unsigned int  size, const char* capture_fn, int ant);

/*! @brief Function to initialize the radio configuration structure to default
 *         values for tradio_test.
 *
 * @returns Nothing (void function)
 *
 * This code sets up the following as defaults:
 *   - 1 antenna (SISO)
 *   - Sampling rate of 15360 I/Q samples per millisecond (10 MHz)
 *   - Band 7
 *   - JESD radio interface as Pulse mode
 *   - Map Sync Mode disabled
 *   - TX and RX Log buffer sizes of 0x5DC000 each
 *   - Receiver gain control as automatic
 *
 */
void RadCfgDefInitialization()
{
    memset (&RadConfig, 0, sizeof (RadConfig));

    RadConfig.cfg4g.NumAntennas           = 1;
    RadConfig.cfg4g.SamplingRate          = SAMPLING_RATE_10MHZ;
    RadConfig.cfg4g.TxLoopback            = 0;
    RadConfig.cfg4g.RxLoopback            = 0;
    RadConfig.cfg4g.Band                  = 7;
    RadConfig.cfg4g.RadioMode             = RADMODE_PULSE;
    RadConfig.cfg4g.MapSyncMode           = MAPSYNCMODE_OFF;
    RadConfig.cfg4g.TxPlayBufUserVirtAddr = 0x00000000;
    RadConfig.cfg4g.TxPlayBufVirtAddr     = 0x00000000;
    RadConfig.cfg4g.TxPlayBufPhysAddr     = 0x00000000;
    RadConfig.cfg4g.TxPlayBufSize         = 0x00000000;
    RadConfig.cfg4g.TxLogBufVirtAddr      = 0x00000000;
    RadConfig.cfg4g.TxLogBufPhysAddr      = 0x00000000;
    RadConfig.cfg4g.TxLogBufSize          = 0x005DC000;
    RadConfig.cfg4g.RxLogBufVirtAddr      = 0x00000000;
    RadConfig.cfg4g.RxLogBufPhysAddr      = 0x00000000;
    RadConfig.cfg4g.RxLogBufSize          = 0x005DC000;
    RadConfig.cfg4g.pRadProfile           = 0;
    RadConfig.cfg4g.MapBootReq            = 1;
    RadConfig.cfg4g.MapBufHeap            = 0;
    RadConfig.cfg4g.RadTimerIrqTrigger    = 0;
    RadConfig.cfg4g.RadFrameSyncEnable    = 0;
    RadConfig.cfg4g.RouteTtiIrqToCeva     = 0;
    RadConfig.cfg4g.GainControl           = RX_GAIN_CNTRL_AGC;
    RadConfig.cfg4g.AuxDacValue           = 0;
    RadConfig.cfg4g.NoMapConfig           = 0;
    RadConfig.cfg4g.NoSyncnetConfig       = 0;
}

/*! @brief Function to setup libradio External Radio chipset library for
 *         use by the tradio_test test program
 *
 * @param[in]  RadioId   libradio external Radio chipset ID number.
 *                       NOTE: this may be different than radio interace ID number
 *
 * @returns  If sucessful, returns pointer to libradio data context structure
 *           used for all subsequent libradio calls
 *
 * Code flow:
 *  - Call libradio Open function (radioOpen) and get context pointer
 *  - Setup libradio debug level (what output messages it prints out).
 *  - Select radio baed on passed radio ID
 *  - On error, and open OK, then close libradio
 *  - Return context pointer if OK, or NULL pointer if failed
 */
static RadioContext* SetupLibradio(int RadioId)
{
    RadioContext * ctxt;
    RadioReturn    err;
    //
    // Open libradio
    // for radio chipset configuration and operational control
    //
    printf("libradio: radioOpen(): ");
    ctxt = radioOpen();
    if (ctxt != NULL) {
        printf("OK\n");
        //
        // libradio Open worked OK
        // First setup debug level
        //
        radioSetDebugLevel(ctxt, radio_debug);
    }
    else {
        printf("FAILED!  SPI BUS MAY ALREADY BE IN USE.\n");
    }
    //
    // If running libradio
    // Use radio Select to select radio prior
    // to other libradio operations
    //
    if ((ctxt != NULL)) {
        //
        // Next setup for subsequent operations
        // Select radio based on Radio ID
        //
        err = radioSelect(ctxt, RadioId);
        if (err != RADIO_ESUCCESS) {
            printf("ERROR %u: radioSelect ID: %u Failed\n",err,RadId);
            radioClose(ctxt);
            ctxt = NULL;
        }
    }
    return ctxt;
}

static void pabort(const char *s)
{
    if (errno) {
        perror(s);
        exit(errno);
    } else {
        printf(s);
        exit(-1);
    }
}

int sig_term_wait(void)
{
    int             sig, ret;
    sigset_t        sigmask;

    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGTERM);
    ret = sigprocmask(SIG_BLOCK, &sigmask, NULL);
    if (ret != 0)
        fprintf(stderr, "Signal arrangements failed %i\n", ret);

    ret = sigwait(&sigmask, &sig);
    if (ret != 0)
        fprintf(stderr, "Waiting failed %i\n", ret);

    return ret;
}


int asu_radio_init()
{
    int          ret = 0;
    int          fd_spi1;
    int          fd_rad;
    int          i;
    int          sampling_rate_index;
    unsigned int val;
    unsigned int rad_tx_dma_buf;
    unsigned int rad_rx_dma_buf;
    unsigned int offset, prev_offset, delta;
    unsigned int log_control = LOG_TX0_ENABLED | LOG_TX1_ENABLED | LOG_RX0_ENABLED | LOG_RX1_ENABLED;
    unsigned int log_size;
    unsigned int radio_test_tx_bytes, radio_test_rx_bytes;
    unsigned int script_cmdbufsize;
    unsigned int sample_iqbufsize;
    unsigned int chip_counter;
    unsigned int frame_counter;
    int          radtype = RADTYPE_NOT_DET;
    char         s[80];

    radio_change_requested = 1;
    //
    // Libradio variables
    //
    RadioContext * ctxt;    // Libradio context (from RadioOpen)
    RadioReturn    err;     // Libradio return

    //
    // Setup default values for local Radio configuration data structure
    //
    RadCfgDefInitialization();
    //Ad936xSetTxFreq(2535);
    //Ad936xSetRxFreq(2655);
    //
    // Sanity check parameters
    // And ajust radio configuration and operational values
    // as necessary, abort for any serious errors
    //

    // Init instead of parse_opts

    printf("in radio_init\n");
    RadId = 1;
    RadConfig.cfg4g.NumAntennas = 1;
    radio_init = 1;
    radio_loopback = 0;
    /* 3840 (3), 7680 (5), 15360 (10), 23040 (15), or 30720 (20) */
     radio_sampling_rate = 3; // to do: change this as an input of the radio init function
    radio_mode = 0; // to do: change this to input
//    RadConfig.cfg4g.MapBootReq = 0; // read more on this

    radio_band = 7; // make this as an input to config
 //   RadConfig.cfg4g.AuxDacValue
    RadConfig.cfg4g.MapSyncMode = MAPSYNCMODE_OFF;


    gpio_rf_sync = 0;
    


    printf("num ant %d\n",RadConfig.cfg4g.NumAntennas);
    if (RadId != 0 && RadId != 1)
        pabort("Invalid RadId ID. Use 0 for CMOS, 1 for LVDS radio\n");
    RadConfig.cfg4g.RadId = RadId;

    if (RadConfig.cfg4g.NumAntennas != 1 && RadConfig.cfg4g.NumAntennas != 2)
        pabort("Invalid number of antennas. Use 1 or 2\n");

    if (RadConfig.cfg4g.AuxDacValue > 0x3FF)
        pabort("Invalid 10 bit AuxDAC value too large, use 1-1023 (0x1-0x3ff)\n");

    if (radio_mode == -1) {
        radio_mode = RADMODE_PULSE; // set Pulse mode as default
    }
    RadConfig.cfg4g.RadioMode = (radio_mode == RADMODE_LEVEL) ? RADMODE_LEVEL : RADMODE_PULSE;

    if (radio_band == -1)
        radio_band = (RadId == 0) ? 2 : 7; // set Band 2 as default for CMOS and Band 7 as default for LVDS
    RadConfig.cfg4g.Band = radio_band;
    printf("band is %d\n",RadConfig.cfg4g.Band);

    if (radio_sampling_rate == -1) {
        radio_sampling_rate = (RadId == 0)
                            ? SAMPLING_RATE_5MHZ  // set  7680 as default rate for radio ID 0 (CMOS)
                            : SAMPLING_RATE_10MHZ // set 15360 as default rate for radio id 1 (LVDS)
                            ;
    }
    else if (radio_sampling_rate == 3) {
        radio_sampling_rate = SAMPLING_RATE_3MHZ;
    }
    else if (radio_sampling_rate == 5) {
        radio_sampling_rate = SAMPLING_RATE_5MHZ;
    }
    else if (radio_sampling_rate == 10) {
        radio_sampling_rate = SAMPLING_RATE_10MHZ;
    }
    else if (radio_sampling_rate == 15) {
        radio_sampling_rate = SAMPLING_RATE_15MHZ;
    }
    else if (radio_sampling_rate == 20) {
        radio_sampling_rate = SAMPLING_RATE_20MHZ;
    }
    else if (   radio_sampling_rate != SAMPLING_RATE_3MHZ
             && radio_sampling_rate != SAMPLING_RATE_5MHZ
             && radio_sampling_rate != SAMPLING_RATE_10MHZ
             && radio_sampling_rate != SAMPLING_RATE_15MHZ
             && radio_sampling_rate != SAMPLING_RATE_20MHZ
            ) {
            printf("ERROR unsupported sampling rate specified: %u\n",radio_sampling_rate);
            pabort("Invalid sampling rate. Supports only rates for 3, 5, 10, 15 or 20 MHz bandwidths");
    }
    //
    // If running libradio, setup libardio configuration
    // and mode options
    //
    //STATRT LIBRADIIO PART
    if (use_libradio) {
        //
        // Running libradio, set SISO or MIMO based
        // on number of antennas requested
        //
        if (RadConfig.cfg4g.NumAntennas == 2) {
            // 2 antennas, request MIMO mode
            libradio_conf = RADIO_CONF_MIMO;
        } else {
            // Not 2 antennas, request SISO mode
            libradio_conf = RADIO_CONF_SISO;
        }
        //
        // Setup libradio mode based on band, radio
        // mode and sampling rate
        //
        if ((radio_band == 1) || (radio_band == 2)) {
            // 3G radio type (based on band)
            libradio_mode = RADIO_MODE_3GNODEB;
        } else {
            if (radio_mode == 2) {
                // TDD mode
                switch(radio_sampling_rate) {
                    case SAMPLING_RATE_5MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_TDD_5MHZ;
                        break;
                    case SAMPLING_RATE_10MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_TDD_10MHZ;
                        break;
                    case SAMPLING_RATE_20MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_TDD_20MHZ;
                        break;
                }
            } else {
                // FDD mode
                switch(radio_sampling_rate) {
                    case SAMPLING_RATE_3MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_FDD_3MHZ;
                        break;
                    case SAMPLING_RATE_5MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_FDD_5MHZ;
                        break;
                    case SAMPLING_RATE_10MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_FDD_10MHZ;
                        break;
                    case SAMPLING_RATE_15MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_FDD_15MHZ;
                        break;
                    case SAMPLING_RATE_20MHZ:
                        libradio_mode = RADIO_MODE_4GENODEB_FDD_20MHZ;
                        break;
                }
            }
        }
    }
// END LIBRADIO
    //
    // Setup TX and RX I/Q data log buffer sizes based
    // on number of antennas and sampling rate
    //
    RadConfig.cfg4g.TxLogBufSize *= RadConfig.cfg4g.NumAntennas;
    RadConfig.cfg4g.RxLogBufSize *= RadConfig.cfg4g.NumAntennas;
    if (radio_sampling_rate == SAMPLING_RATE_5MHZ)
    {
        sampling_rate_index     = 0;
        RadConfig.cfg4g.TxLogBufSize /= 2;
        RadConfig.cfg4g.RxLogBufSize /= 2;
    }
    else if (radio_sampling_rate == SAMPLING_RATE_10MHZ)
        sampling_rate_index = 1; // sampling rate index of 1 for 10 MHz
    else
        sampling_rate_index = 2; // sampling rate index of 2 for 15 and 20 MHz

    RadConfig.cfg4g.SamplingRate = radio_sampling_rate;

    if (radio_loopback == 0) {
        RadConfig.cfg4g.TxLoopback = 0;
        RadConfig.cfg4g.RxLoopback = 0;
    }
    else if (radio_loopback == 1)
        RadConfig.cfg4g.TxLoopback = 1;
    else if (radio_loopback == 2)
        RadConfig.cfg4g.RxLoopback = 1;
    else
        pabort("Invalid radio loopback mode. Supports only 0-no loopback, 1-Tx loopback and 2-Rx loopback\n");

    if (RadConfig.cfg4g.MapSyncMode > 2)
        pabort("Invalid MAP sync mode. Supports only 0-no MAP, 1-4G sync and 2-3G sync\n");

    //
    // Open libradio if requested and radio programming/change needed
    // for radio chipset configuration and operational control
    //
    if ((radio_change_requested) && (use_libradio))
        ctxt = SetupLibradio(RadId);
    else
        ctxt = NULL;

    if (radio_reset_request)
    {
        // TEST, USE LIBRADIO IF OPENED OK
        if (ctxt != NULL)
        {
            // LIBRADIO OPENED OK
            // USE LIBRADIO radio reset function instead
            // of tradio
            printf("libradio: radioReset(): ");
            err = radioReset(ctxt);
            if (err != RADIO_ESUCCESS) {
                printf("ERROR %u\n",err);
                radioClose(ctxt);
                ctxt = NULL;
            } else {
                printf("OK\n");
            }
        } else {
            // LIBRADIO NOT opened or requested, USE TRADIO RESET
            RadioReset();
        }
    }

    if (ctxt == NULL)
    {
        // To not interfere with other programs
        // such as msradioapp, only probe the radio
        // if one of the parameters setup for
        // tradio_test needs to do an operation
        // to the TransRF (Ducatti) or AD936X chipset
        if (radio_change_requested) {
//            radtype = RadioProbe();
radtype = RADTYPE_AD9361_RFC;
            if (radtype == RADTYPE_NOT_DET) {
                // Probe failed to detect radio type, exit program
                return 0;
            }
        }
    }
    // Test if radtype detected on above code, if not
    // assume Ducatti if -D option set and AD936X if not
    if (radtype == RADTYPE_NOT_DET) {
        if (ducatti_cpri_rate)
            radtype = RADTYPE_DUCATTI;
        else
            radtype = RADTYPE_AD9361_RFC;
    }
    if (script_init_fn) {
        if (ctxt && use_libradio) {
            // libradio, use libradio script init function
            //
            // LIBRADIO INIT BY SCRIPT SELECTED
            //
            printf("\nlibradio radioInitByScript config:%u, mode:%u, band:%u script_init_fn:%s ",
                   libradio_conf,
                   libradio_mode,
                   radio_band,
                   script_init_fn
                   );

            err = radioInitByScript
                             (ctxt,           // RadioContext* ctxt,
                              libradio_conf,  // RadioConf     conf,
                              libradio_mode,  // RadioMode     mode,
                              radio_band,     // RadioBand     band,
                              script_init_fn  //const char*   script_init_fn
                             );
            if (err != RADIO_ESUCCESS) {
                printf("ERROR %u\n",err);
                radioClose(ctxt);
                ctxt = NULL;
            } else {
                printf("OK\n");
            }
        } else {
            // Not libradio, use libtradio
            RadConfig.cfg4g.pRadProfile = (void *)ADPARSERProccessScript(script_init_fn);
        }
    }

    if (radio_init) {
        if (ctxt && use_libradio) {
            //
            // LIBRADIO INIT SELECTED
            //
            printf("\nlibradio radioInit config:%u, mode:%u, band:%u ",
                   libradio_conf,
                   libradio_mode,
                   radio_band
                   );
            err = radioInit(ctxt,
                            libradio_conf,
                            libradio_mode,
                            radio_band
                           );
            if (err == RADIO_ESUCCESS)
                printf("OK\n");
            else {
                printf("\ERROR %u\n",err);
                //
                // Close radio
                //
                radioClose(ctxt);
                ctxt = NULL;
             }
        } else {
        // Not running libradio and radio initialization requested
        // use libtradio RadioInit
            if (RadioInit(RadId, (PRADCONF)&RadConfig) < 0)
                return 0;
        }
    } else {
        //
        // If not radio initialize not performed, (which will
        // update the AuxDac if non-zero value), now check if
        // AUXDAC value needs to be updated without radio initialize,
        // If so, then update radio chipset's AuxDac value
        // to adjust radio reference oscillator frequency
        //
        // TODO: This code is currently hard coded here for AD936X
        // for simplicity, to allow easy patch in older
        // versions of software and also to minimize risk
        // of the change.  Need to add Ducatti support, more flexibility
        // etc. (work in progress, not fully tested)
        //
        if (    (radtype == RADTYPE_AD9361_RFC)
            && (RadConfig.cfg4g.AuxDacValue != 0)
            && (use_libradio == 0)
           ) {
            Ad9361Write(1,0x18,(RadConfig.cfg4g.AuxDacValue >> 2));           // Set AuxDAC bits 9:2
            Ad9361Write(1,0x1A,((RadConfig.cfg4g.AuxDacValue & 3) | 0x14));   // Set AuxDac bits 1:0 and control bits
        }
    }

    if ((RadConfig.cfg4g.AuxDacValue != 0) && (ctxt != NULL)) {
        //
        // Aux Dac value change requested and running libradio
        // For test purposes, validate DAC width OK and that
        // if OK, requested DAC value is within proper range
        //
        radio_auxdac_width = radioGetOscillatorDacWidth(ctxt);
        radio_auxdac_max   = (1 << radio_auxdac_width) - 1;
        if (RadConfig.cfg4g.AuxDacValue > radio_auxdac_max) {
            printf("ERROR: AuxDAC value requested: %u, Max:%d\n", RadConfig.cfg4g.AuxDacValue, radio_auxdac_max);
        } else {
            //
            // Dac value request value OK and AuxDac API enabled for this
            // radio.  Setup new Aux DAC value
            //
            printf("Current AuxDacValue: 0x%X\n",radioGetOscillatorDacValue(ctxt));
            printf("Set New AuxDacValue: ");
            err = radioSetOscillatorDacValue(ctxt, (unsigned short)RadConfig.cfg4g.AuxDacValue);
            if (err == RADIO_ESUCCESS) {
                //
                // AuxDac value set OK:
                //
                radio_auxdac_value = RadConfig.cfg4g.AuxDacValue;
                printf("0x%X OK\n", radio_auxdac_value);
            } else {
                //
                // AuxDac set failed for some reason
                //
                printf("SET FAILED, ERROR %u\n",err);
                radioClose(ctxt);
                ctxt = NULL;
            }
        }
    }

    if(ducatti_cpri_rate)
    {
        // initialize CPRI mode for ducatti
        CPRIPARAM cpri;
        cpri.Loopback = radio_loopback;
        cpri.Rate = ducatti_cpri_rate;
        cpri.SamlingRate = radio_sampling_rate;
        cpri.Width = 16;

        printf("Switching Ducatti to CPRI interface (rate = %d.%d Mbps)\n",
               6144*ducatti_cpri_rate/10,
               6144*ducatti_cpri_rate%10
              );
        DucattiCpriInit(RadId, &cpri);
    }

    // If libradio, setup loopback option now
    // Note depending on programming mode and chip options
    // Loopback may not be supported
    if (ctxt != NULL) {
        printf("\nlibradio radioSetLoopback() Option: %u ",
               RadConfig.cfg4g.TxLoopback
               );
        err = radioSetLoopback(ctxt,
                               RadConfig.cfg4g.TxLoopback ? RADIO_TX_TO_RX_LOOPBACK
                                                          : RADIO_NO_LOOPBACK
                              );
        if (err == RADIO_ESUCCESS) {
            //
            // Loopback set OK:
            //
            printf("OK\n");
        } else {
            //
            // Loopback set failed for some reason
            //
            printf("ERROR %u\n",err);
            radioClose(ctxt);
            ctxt = NULL;
        }
    }
    //
    // Close libradio if not already closed
    //
    if (ctxt != NULL) {
        radioClose(ctxt);
        ctxt = NULL;
    }
    //
    // Close libtradio (note multiple closes won't hurt for that call)
    // This will release the SPI bus so other applications (such as
    // msradioapp) can use the SPI bus.
    //
    RadioClose();


    printf("%d\n",RadConfig.cfg4g.Band);
    return 0;
}/* end of radio init */


    // BEGIN TX TX TX
int asu_tx_rx( unsigned int input_buffer_add, unsigned int output_buffer_add,
		int radio_test_time)
{
    int          ret = 0;
    int          fd_spi1;
    int          fd_rad;
    int          i;
    int          sampling_rate_index;
    unsigned int val;
    unsigned int rad_tx_dma_buf;
    unsigned int rad_rx_dma_buf;
    unsigned int *p_out_buf;
    unsigned int offset, prev_offset, delta;
    unsigned int log_control = LOG_TX0_ENABLED | LOG_TX1_ENABLED | LOG_RX0_ENABLED | LOG_RX1_ENABLED;
    unsigned int log_size;
    unsigned int radio_test_tx_bytes, radio_test_rx_bytes;
    unsigned int script_cmdbufsize;
    unsigned int sample_iqbufsize;
    unsigned int chip_counter;
    unsigned int frame_counter;
    int          radtype = RADTYPE_NOT_DET;
    char         s[80];
    unsigned int pRxData;
    int 	 rxsize;
    //
    // Libradio variables
    //
    RadioContext * ctxt;    // Libradio context (from RadioOpen)
    RadioReturn    err;     // Libradio return

    RadId = 1;
    radio_loopback = 0;
    radio_mode = 0; // to do: change this to input

    radio_band = 7; 

//printf("radio_test_time = %d\n", radio_test_time);

    if (radio_test_time) {
        // Radio data test requested, open kernel radio device (JESD/JDMA handler)
        // Abort if unable to open the device
        fd_rad = open(rad_device[RadId], O_RDWR);
        if (fd_rad < 0) {
            printf("Can't open %s device. ", rad_device[RadId]);
            abort();
        }
        //
        // Radio JESD/JDMA device opened OK, check
        // if binary sample file specified to run
        // for this test (instead of pre-compiled data)
        //
        if (input_buffer_add != 0) {

         //   RadConfig.cfg4g.TxPlayBufUserVirtAddr = (unsigned int) &sin1mhziqbuf[0]; //&ltedlqam16iqbuf[0] ;//(unsigned int)&sin1mhziqbuf[0];
         //  RadConfig.cfg4g.TxPlayBufSize = 4 * sin1mhziqnum;
         //

/*
{
    int i;
    for(i=0; i<15360; i++) {
        printf("%d %d %d\n", i, txdata[i*2], txdata[i*2+1]);
    }
}
*/

		RadConfig.cfg4g.TxPlayBufUserVirtAddr = (unsigned int)&txdata[0];
		RadConfig.cfg4g.TxPlayBufSize = sizeof(txdata);
//            printf("Setting up data from file for playback...\n");

        }

/*
        printf("Running Radio Test (%s Radio Mode, MAPSyncMode = %d SamplingRate=%d NumAntennas=%d, Loopback=%d) for %d ms...\n",
               (radio_mode == RADMODE_LEVEL) ? "Level" : "Pulse",
               RadConfig.cfg4g.MapSyncMode,
               RadConfig.cfg4g.SamplingRate,
               RadConfig.cfg4g.NumAntennas,
               radio_loopback,
               radio_test_time
              );
*/
        //
        // Send ioctl to configure JESD/JDMA interface based on data
        // stored in RadConfig structure
        //
 
        ret = ioctl(fd_rad, TRAD_IOCTL_SET_RAD_CONFIG, &RadConfig);
/*
        printf("MapDlInBufPhysAddr[0]=%x MapDlInBufPhysAddr[1]=%x MapUlOutBufPhysAddr[0]=%x MapUlOutBufPhysAddr[1]=%x\n",
               RadConfig.cfg4g.MapDlInBufPhysAddr[0],
               RadConfig.cfg4g.MapDlInBufPhysAddr[1],
               RadConfig.cfg4g.MapUlOutBufPhysAddr[0],
               RadConfig.cfg4g.MapUlOutBufPhysAddr[1]
              );
*/
        if (ret)
            pabort("Radio Config IOCTL failed\n");

        ret = ioctl(fd_rad, TRAD_IOCTL_GET_TX_BUF_PTR, &rad_tx_dma_buf);
        ret = ioctl(fd_rad, TRAD_IOCTL_GET_RX_BUF_PTR, &rad_rx_dma_buf);
        ret = ioctl(fd_rad, TRAD_IOCTL_LOG_CONTROL,    &log_control);
        ret = ioctl(fd_rad, TRAD_IOCTL_RFSYNC,         &gpio_rf_sync);

        log_size = RadConfig.cfg4g.RxLogBufSize;
        if (RadConfig.cfg4g.NumAntennas == 2)
            log_size /= 2;

/*
        printf("TxPlayBufPA=0x%x TxPlayBufSize=0x%x\n",
               RadConfig.cfg4g.TxPlayBufPhysAddr,
               RadConfig.cfg4g.TxPlayBufSize
              );

        printf("Tx0LogPA=0x%x Rx0LogPA=0x%x\n",
               RadConfig.cfg4g.TxLogBufPhysAddr,
               RadConfig.cfg4g.RxLogBufPhysAddr
              );
        if (RadConfig.cfg4g.NumAntennas == 2) {
            printf("Tx1LogPA=0x%x Rx1LogPA=0x%x\n",
                   RadConfig.cfg4g.TxLogBufPhysAddr + log_size,
                   RadConfig.cfg4g.RxLogBufPhysAddr + log_size
                  );
        }
        printf("Tx0DmaPA=0x%x Rx0DmaPA=0x%x\n",
               rad_tx_dma_buf,
               rad_rx_dma_buf
              );
        if (RadConfig.cfg4g.NumAntennas == 2) {
            if(RadConfig.cfg4g.MapSyncMode == 0)
                printf("Tx1DmaPA=0x%x Rx1DmaPA=0x%x\n",
                       rad_tx_dma_buf+8*RadConfig.cfg4g.SamplingRate,
                       rad_rx_dma_buf+8*RadConfig.cfg4g.SamplingRate
                      );
            else
                printf("Tx1DmaPA=0x%x Rx1DmaPA=0x%x\n",
                       rad_tx_dma_buf+2*RadConfig.cfg4g.SamplingRate,
                       rad_rx_dma_buf+2*RadConfig.cfg4g.SamplingRate
                  );
        }
*/

        RadioTimerOpen();

        if (radio_sync_start) {
            if (RadId == 1)
                ioctl(fd_rad, TRAD_IOCTL_ENABLE_SYNC, 0);
        } else {
            ret = ioctl(fd_rad, TRAD_IOCTL_ENABLE, 0);
            if (ret) {
                printf("Radio enable failed.\n");
            }
            if(RadConfig.cfg4g.NoMapConfig == 1)
            {
                ret = sig_term_wait();

                ret = ioctl(fd_rad, TRAD_IOCTL_ENABLE, 0);
            }
        }
        //
        // Test if running 3G MAP Sync mode (2)
        //
        if (RadConfig.cfg4g.MapSyncMode == MAPSYNCMODE_3G) {
            //
            // Running 3G Map Syn mode, test if running
            // Radio Timer Interrupt Trigger
            //
            if (RadConfig.cfg4g.RadTimerIrqTrigger) {
                //
                // Running 3G mode and running radio timer interrupt trigger
                // Loop calling ioctl to get Current RX buffer offset
                // until test time expired
                //
                for (i = 0; i < radio_test_time/RadConfig.cfg4g.RadTimerIrqTrigger; i++) {
                    ret = ioctl(fd_rad, TRAD_IOCTL_GET_CUR_RX_OFFSET, &offset);
                    printf("%d\n", i);
                }
            } else {
                //
                // Running 3G, but not using radio timer interrupt trigger.
                //
#ifdef RADTM_TEST_ENABLE
                //
                // If RADTM_TEST_ENABLE compile option set, then
                // test time loop run by querying the
                // radio timer.
                //
                for (i = 0; i < radio_test_time; i++) {
                    usleep(1000);
                    RadioTimerQuery(&chip_counter, &frame_counter);
                    if (i < sizeof(rad_timer_records)/sizeof(rad_timer_records[0]))
                        rad_timer_records[i] = (frame_counter<<16) | chip_counter;
                }
#else
                //
                // If RADTM_TEST_ENABLE compile option not set, then
                // call usleep function to sleep for requested number
                // of milliseconds for the test loop.
                //
                usleep(radio_test_time*1000);
#endif
            }
        //
        // Not running 3G MAP sync mode, test if running
        // 4G MAP Sync mode
        //
        } else if (RadConfig.cfg4g.MapSyncMode == MAPSYNCMODE_4G) {
            //
            // Running 3G MAP Sync mode:
            // call usleep function to sleep for requested number
            // of milliseconds for the test loop.
            //
            usleep(radio_test_time*1000);
        } else {
            //
            // Not running MAP Sync mode (3G or 4G)
            // Call TRAD_IOCTL_GET_CUR_RX_OFFSET to
            // wait for each time and RX offset is
            // ready to process.
            //
            // For IRQ test, if 15 MHz bandwidth
            // actual JESD/JDMA sampling rate is 20 MHz (30720 samples/millisecond)
            //
            if (radio_sampling_rate == SAMPLING_RATE_15MHZ)
                radio_sampling_rate =  SAMPLING_RATE_20MHZ;
            for (i = 0; i < radio_test_time; i++) {
                ret = ioctl(fd_rad, TRAD_IOCTL_GET_CUR_RX_OFFSET, &offset);
                //printf("%d\n", offset);
                if (i) {
                    delta = (offset > prev_offset) ? offset - prev_offset : prev_offset - offset;
                    if (delta != (4*radio_sampling_rate))
                        printf("Missed IRQ. delta=%d (0x%x)\n", delta, delta);
                }

                prev_offset = offset;
            }
        }
        //
        // Radio test loop completed, setup for end of test
        // First check if necessary to reopen the SPI interface
        // to be able to send additional commands to the external
        // radio chipset to start shutting things down
        // and/or getting any data from the chipset
        //
        if (radtype != RADTYPE_DUCATTI) {
            if (   (radio_mode == RADMODE_LEVEL)
                || (radio_mode == RADMODE_TDD_PULSE)
               ) {
                   if (ctxt == NULL) {
                       printf("Setting up libradio for post test loop radio changes\n");
                       ctxt = SetupLibradio(RadId);
                   }
            }
        }
        if (   (radtype   != RADTYPE_DUCATTI)
            && (radio_mode == RADMODE_TDD_PULSE)
           ) {
            if (ctxt != NULL) {
                // use libradio to change state of AD936X for TDD pulse mode
                printf("Setting up libradio for post test loop TDD Pulse mode radio changes\n");
                radioSpiWrite(ctxt,
                              0x15,
                              (radioSpiRead(ctxt,0x15) & 0x7F)
                             );
                radioSpiWrite(ctxt,
                              0x14,
                              0x23
                             );
            } else {
                // use tradio to change state of AD936X for TDD pulse mode
                Ad9361Write(RadId, 0x15, Ad9361Read(RadId, 0x15) & 0x7F);
                Ad9361Write(RadId, 0x14, 0x23);
            }
        }
        //
        // Build and send and ioctl to disable the JESD or CPRI
        // radio interface
        //
        ret = ioctl(fd_rad, TRAD_IOCTL_DISABLE, 0);
        //
        // If necessary change start in external radio chipset to an idle state
        //
        if (radtype != RADTYPE_DUCATTI) {
            if (   (radio_mode == RADMODE_LEVEL)
                || (radio_mode == RADMODE_TDD_PULSE)
               ) {
                if (ctxt != NULL) {
                    // use libradio
                    // for level mode move radio to ALERT state
                    radioSpiWrite(ctxt,
                                  0x14,
                                  0x03
                                 );
                } else {
                    // use tradio
                    // for level mode move radio to ALERT state
                    Ad9361Write(RadId, 0x14, 0x03); // for level mode move radio to ALERT state
                }
            }
        }
#ifdef RADTM_TEST_ENABLE
        //
        // If running radio timer test, then display radio timer debug information
        //
        if (RadConfig.cfg4g.MapSyncMode && RadConfig.cfg4g.RadTimerIrqTrigger == 0) {
            for (i = 0; i < radio_test_time; i++)
                if (i < sizeof(rad_timer_records)/sizeof(rad_timer_records[0]))
                    printf("Frame: %d Chip: %d\n",
                           rad_timer_records[i] >> 16,
                           rad_timer_records[i] & 0xFFFF
                          );
        }
#endif
        //
        // Calculate total and log sizes for number of bytes transmitted
        // and received
        //
        radio_test_tx_bytes = radio_test_time * RadConfig.cfg4g.SamplingRate * 4;
        if (RadConfig.cfg4g.TxLogBufSize > radio_test_tx_bytes)
            RadConfig.cfg4g.TxLogBufSize = radio_test_tx_bytes;
        radio_test_rx_bytes = radio_test_time * RadConfig.cfg4g.SamplingRate * 4;
        if (RadConfig.cfg4g.RxLogBufSize > radio_test_rx_bytes)
            RadConfig.cfg4g.RxLogBufSize = radio_test_rx_bytes;

	rxsize = RadConfig.cfg4g.RxLogBufSize;
        if (ret) {
            printf("ERROR: Radio disable failed.\n");
        } else {
            //
            // Test if running 1 or two antennas
            //
            if (RadConfig.cfg4g.NumAntennas == 1) {
		    if(output_buffer_add) {
			    pRxData = (unsigned int) p2vconv(RadConfig.cfg4g.RxLogBufPhysAddr, rxsize);
			    memmove((void*)output_buffer_add, (void*)(pRxData), rxsize);
		    }
               /* if(capture_fn) {
                    SaveCapturedSignal(RadConfig.cfg4g.RxLogBufPhysAddr, rxsize, capture_fn, 0);
                }*/

                }

            }
        //
        // Close radio interface kernel driver
        //
        close(fd_rad);
        if (script_init_fn)
        {
            if (RadConfig.cfg4g.pRadProfile)
            {
                free(RadConfig.cfg4g.pRadProfile);
                RadConfig.cfg4g.pRadProfile = NULL;
            }
        }
    }
    //
    // Close/Shutdown the radio timer block
    //
    RadioTimerClose();
    //
    // Close libradio if not already closed
    //
    //
    // Close libtradio (note multiple closes won't hurt for that call)
    // This will also do a clean release of the SPI bus
    // rather than a close due to program exit
    //
    RadioClose();
    //
    // Return any error code set (or 0 if OK)
    //
    return ret;
}

static int SaveCapturedSignal(unsigned int pa, unsigned int  size, const char* capture_fn, int ant)
{
    unsigned int pRxData = (unsigned int) p2vconv(pa, size);
    int iq;
    FILE *f;
    char buf[256];
    int i;
    short int *temp;
    temp = (short int *) pRxData;

    if(capture_fn == 0 || size == 0)
        return -1;

    sprintf(buf, "%s.txt", capture_fn);
    f = fopen(buf, "w");
    if(f < 0)
    {
        printf("Cannot open %s file for writing\n", buf);
        return -1;
    }
    printf("before writing to file %hd\n", (*(temp+10)));
    iq = size/4;
    for( i=0; i<iq; i++) {
	    fprintf(f, "%hd %hd\n",(*temp), (*(temp+1)));
	    temp = temp + 2;
    }

    fclose(f);
    printf("Antenna data file: %s (%d IQ samples)\n", buf, size/4);

    return 0;
}

