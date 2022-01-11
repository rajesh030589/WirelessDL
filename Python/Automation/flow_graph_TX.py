#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: FM Receiver
# GNU Radio version: v3.10.0.0git-513-ga6fbb06a

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
<<<<<<< HEAD




class fm_block(gr.top_block):

=======
import os

sys.stdout = open(os.devnull, "w")


class fm_block(gr.top_block):
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
    def __init__(self):
        gr.top_block.__init__(self, "FM Receiver", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
<<<<<<< HEAD
        self.tx_gain = tx_gain = 10
=======
        self.tx_gain = tx_gain = 15
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
        self.samp_rate = samp_rate = 10e6
        self.rx_gain = rx_gain = 25
        self.freq = freq = 2.2e9

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
<<<<<<< HEAD
            ",".join(("addr=192.168.20.2", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
=======
            ",".join(("addr=192.168.10.2", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=list(range(0, 1)),
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)
<<<<<<< HEAD
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/rajesh/ICLRWork/WirelessDL/Python/Automation/TX.bin', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)



=======
        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_gr_complex * 1,
            "/home/rajesh/ICLRWork/WirelessDL/Python/Automation/TX.bin",
            True,
            0,
            0,
        )
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)

>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.uhd_usrp_sink_0, 0))

<<<<<<< HEAD

=======
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)


<<<<<<< HEAD


=======
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
def main(top_block_cls=fm_block, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
<<<<<<< HEAD
    # time.sleep(30)
    tb.wait()

    # tb.stop()
if __name__ == '__main__':
    main()
=======
    # time.sleep(110)
    tb.wait()

    # tb.stop()


if __name__ == "__main__":
    main()
>>>>>>> 86490838c12022ef244da0775f471d09c9cbd639
