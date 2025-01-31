"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class ReadPayloadLength(gr.sync_block):
    def __init__(self, total_bytes=2, input_tag="length start", output_tag="payload start", whitening=True):
        gr.sync_block.__init__(
            self,
            name="Read Payload Length",
            in_sig=[np.uint8],  # Input is binary stream
            out_sig=[np.uint8],  # Pass-through output stream
        )
        self.input_tag_key = gr.pmt.intern(input_tag)  # Tag to search for
        self.output_tag_key = gr.pmt.intern(output_tag)  # Tag to apply on data stream

        self.total_bits = 8 * total_bytes  # Number of bits to read (last byte contains length information)
        self.buffer = np.array([], dtype=np.uint8)  # Buffer to hold data across chunks
        self.buffering_active = False  # Flag to indicate active buffering
        self.pending_tag = None  # Store the tag for processing when buffer is filled

        self.whitening = whitening

    def work(self, input_items, output_items):
        in_data = input_items[0]  # Input stream
        out_data = output_items[0]  # Output stream
        nread = self.nitems_read(0)  # Number of items read so far on input
        tags = self.get_tags_in_window(0, 0, len(in_data))  # Tags in current window

        # Copy input to output (pass-through block)
        out_data[:] = in_data

        # Handle buffering
        if self.buffering_active:
            self.buffer = np.append(self.buffer, in_data)
            if len(self.buffer) >= self.total_bits:  # Check if we have enough bits
                self.process_buffered_data()

        # If not buffering, look for a triggering tag that starts buffering or direct processing if possible within the window
        # Assumes that packets are sent at least one window appart (worst case, 256 symbols)
        else:          
            for tag in tags:
                if tag.key == self.input_tag_key:  # Look for triggering tag
                    tag_pos_window = tag.offset - nread  # Position of the tag in the current window
                    
                    # Check if the required number of bits is within the current chunk
                    if tag_pos_window + self.total_bits <= len(in_data):
                        # Extract the bits directly from the current chunk
                        payload_bits = in_data[tag_pos_window : tag_pos_window + self.total_bits]
                        self.process_payload(tag, payload_bits)
                    else:
                        # Start buffering if data spans multiple chunks
                        self.buffering_active = True
                        self.pending_tag = tag  # Store the tag for later use
                        self.buffer = np.array(in_data[tag_pos_window:], dtype=np.uint8)  # Add available bits to buffer

                    break  # Only analyse first triggering tag (in case there is more than one)

            return len(output_items[0])

    def process_payload(self, tag, payload_bits):
        # Convert bits to decimal
        data_bytes = self.binary_to_uint8_array(payload_bits)
        if self.whitening:
            data_bytes, lfsr = self.BLE_whitening(data_bytes)

        # Add a new tag for the start of the payload
        self.add_item_tag(
            0,  # Output port
            tag.offset + self.total_bits,  # Absolute position of the payload start
            self.output_tag_key,
            gr.pmt.init_u8vector(2, bytearray([data_bytes[-1], lfsr])),  # Length of the payload
        )

    def process_buffered_data(self):
        # Process the buffered data once it has enough bits
        payload_bits = self.buffer[:self.total_bits]
        self.process_payload(self.pending_tag, payload_bits)

        # Empty the buffer and reset state
        # self.buffer = np.array([], dtype=np.uint8)
        self.buffering_active = False
        self.pending_tag = None
    
    def BLE_whitening(self, data, lfsr=0x01, polynomial=0x11):
        # Input data must be LSB first (0x7C = 124 must be formatted as 0011 1110)
        # LFSR default value is 0x01 as it is the default value in the nRF DATAWHITEIV register
        # The polynomial default value is 0x11 = 0b001_0001 -> x⁷ + x⁴ + 1 (x⁷ is omitted)
        output = np.empty_like(data)  # Initialise output array

        for idx, byte, in enumerate(data):
            whitened_byte = 0
            for bit_pos in range(8):
                # XOR the current data bit with LFSR MSB
                lfsr_msb = (lfsr & 0x40) >> 6  # LFSR is 7-bit, so MSB is at position 6
                data_bit = (byte >> bit_pos) & 1  # Extract current bit
                whitened_bit = data_bit ^ lfsr_msb  # XOR
                
                # Update the whitened byte
                whitened_byte |= (whitened_bit << (bit_pos))

                # Update LFSR
                lfsr = (lfsr << 1) & 0x7F  # 0x7F mask to keep ksfr within 7 bits
                if lfsr_msb:  # If MSB is 1 (before shifting), apply feedback
                    lfsr = lfsr ^ polynomial  # XOR with predefined polynomial

            output[idx] = whitened_byte

        return output, lfsr
    
    def binary_to_uint8_array(self, binary):       
        # Ensure the binary array length is a multiple of 8
        if len(binary) % 8 != 0:
            raise ValueError("The binary list length must be a multiple of 8.")
        
        # Convert binary to NumPy array
        binary_array = np.array(binary, dtype=np.uint8)
        binary_array = binary_array.reshape(-1, 8)[:, ::-1] #  LSB to MSB correction
        uint8_array = np.packbits(binary_array, axis=1).flatten()

        return uint8_array
