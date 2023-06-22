#ifndef __RF_INTERFACE_H__
#define __RF_INTERFACE_H__

#define TXMODE 0
#define RXMODE 1

#include "p2vconv.h"

int asu_radio_init();
int asu_tx_rx(unsigned int input_buffer_add, unsigned int output_buffer_add, int radio_test_time);

#endif
