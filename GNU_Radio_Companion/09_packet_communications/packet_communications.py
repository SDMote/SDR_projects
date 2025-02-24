#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: packet_communications
# Author: Barry Duggan
# Description: Baseband packet
# GNU Radio version: v3.11.0.0git-843-g6b25c171

from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
import threading




class packet_communications(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "packet_communications", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.num_key = num_key = "packet_num"
        self.len_key = len_key = "packet_len"
        self.samp_rate = samp_rate = 32e3
        self.hdr_format = hdr_format = digital.header_format_crc(len_key, num_key)

        ##################################################
        # Blocks
        ##################################################

        self.pdu_tagged_stream_to_pdu_0 = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        self.pdu_pdu_to_tagged_stream_0_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.digital_protocol_parser_b_0 = digital.protocol_parser_b(hdr_format)
        self.digital_protocol_formatter_async_0 = digital.protocol_formatter_async(hdr_format)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            32,
            1,
            0,
            "packet_len",
            "packet_len",
            False,
            gr.sizeof_char,
            "burst",
            int(samp_rate),
            (),
            0)
        self.digital_crc_check_0 = digital.crc_check(8, 0x4C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True, False, False, 0)
        self.digital_crc_append_0 = digital.crc_append(8, 0x4C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True, False, 0)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_char*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char*1, 'packet_len', 0)
        self.blocks_tag_debug_0 = blocks.tag_debug(gr.sizeof_char*1, '', "")
        self.blocks_tag_debug_0.set_display(True)
        self.blocks_repack_bits_bb_1_0 = blocks.repack_bits_bb(8, 1, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_1 = blocks.repack_bits_bb(1, 8, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.cons(pmt.PMT_NIL,pmt.init_u8vector(9,(71,78,85,32,82,97,100,105,111))), 2000)
        self.blocks_message_debug_0_0 = blocks.message_debug(True, gr.log_levels.debug)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.digital_crc_append_0, 'in'))
        self.msg_connect((self.digital_crc_append_0, 'out'), (self.digital_protocol_formatter_async_0, 'in'))
        self.msg_connect((self.digital_protocol_formatter_async_0, 'header'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.digital_protocol_formatter_async_0, 'payload'), (self.pdu_pdu_to_tagged_stream_0_0, 'pdus'))
        self.msg_connect((self.digital_protocol_parser_b_0, 'info'), (self.blocks_message_debug_0_0, 'print'))
        self.msg_connect((self.digital_protocol_parser_b_0, 'info'), (self.digital_header_payload_demux_0, 'header_data'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0, 'pdus'), (self.digital_crc_check_0, 'in'))
        self.connect((self.blocks_repack_bits_bb_1, 0), (self.blocks_tag_debug_0, 0))
        self.connect((self.blocks_repack_bits_bb_1, 0), (self.pdu_tagged_stream_to_pdu_0, 0))
        self.connect((self.blocks_repack_bits_bb_1_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.blocks_repack_bits_bb_1_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.blocks_repack_bits_bb_1, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.digital_protocol_parser_b_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0_0, 0), (self.blocks_tagged_stream_mux_0, 1))


    def get_num_key(self):
        return self.num_key

    def set_num_key(self, num_key):
        self.num_key = num_key
        self.set_hdr_format(digital.header_format_crc(self.len_key, self.num_key))

    def get_len_key(self):
        return self.len_key

    def set_len_key(self, len_key):
        self.len_key = len_key
        self.set_hdr_format(digital.header_format_crc(self.len_key, self.num_key))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)

    def get_hdr_format(self):
        return self.hdr_format

    def set_hdr_format(self, hdr_format):
        self.hdr_format = hdr_format




def main(top_block_cls=packet_communications, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
