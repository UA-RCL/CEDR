#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.4.0

from gnuradio import blocks
import pmt
from gnuradio import cedr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import zeromq




class range_doppler_gr_cedr(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.vec_len = vec_len = 128
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pub_sink_0 = zeromq.pub_sink(gr.sizeof_gr_complex, vec_len, 'tcp://127.0.0.1:50002', 100, False, (-1), '')
        self.cedr_fft_0_0_0 = cedr.fft(True, vec_len)
        self.cedr_fft_0_0 = cedr.fft(False, vec_len)
        self.cedr_fft_0 = cedr.fft(False, vec_len)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(vec_len)
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_gr_complex*vec_len, 'rx_pulses.bin', True, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*vec_len, 'matched_filter.bin', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.cedr_fft_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.cedr_fft_0_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.cedr_fft_0_0_0, 0))
        self.connect((self.cedr_fft_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.cedr_fft_0_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.cedr_fft_0_0_0, 0), (self.zeromq_pub_sink_0, 0))


    def get_vec_len(self):
        return self.vec_len

    def set_vec_len(self, vec_len):
        self.vec_len = vec_len

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(top_block_cls=range_doppler_gr_cedr, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
