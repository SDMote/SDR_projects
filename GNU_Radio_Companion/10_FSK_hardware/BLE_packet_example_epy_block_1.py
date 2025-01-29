"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class ReadPayload(gr.sync_block):
    def __init__(self, input_tag="payload start", output_tag="CRC end", CRC_size=3):
        gr.sync_block.__init__(
            self,
            name="Read Payload",
            in_sig=[np.uint8],  # Input is binary stream
            out_sig=[np.uint8],  # Pass-through output stream
        )
        self.input_tag_key = gr.pmt.intern(input_tag)  # Tag to search for
        self.output_tag_key = gr.pmt.intern(output_tag)  # Tag to apply on data stream
        self.CRC_size = CRC_size  # CRC size in bytes


        self.num_bits = 0  # Number of bits to read
        self.buffer = []  # Buffer to hold data across chunks
        self.buffering_active = False  # Flag to indicate active buffering
        self.pending_tag = None  # Store the tag for processing when buffer is filled

    def work(self, input_items, output_items):
        in_data = input_items[0]  # Input stream
        out_data = output_items[0]  # Output stream
        nread = self.nitems_read(0)  # Number of items read so far on input
        tags = self.get_tags_in_window(0, 0, len(in_data))  # Tags in current window

        # Copy input to output (pass-through block)
        out_data[:] = in_data

        # Handle buffering
        if self.buffering_active:
            self.buffer.extend(in_data)
            if len(self.buffer) >= self.num_bits:  # Check if we have enough bits
                self.process_buffered_data()
            else:
                return len(output_items[0])
                   
        # If not buffering, look for a triggering tag that starts buffering or direct processing if possible within the window
        for tag in tags:
            if tag.key == self.input_tag_key:  # Look for triggering tag
                tag_pos_window = tag.offset - nread  # Position of the tag in the current window

                self.num_bytes = (gr.pmt.to_long(tag.value) + self.CRC_size) # Number of bytes to read
                self.num_bits = 8 * self.num_bytes # Number of bits to read

                # Check if the required number of bits is within the current chunk
                if tag_pos_window + self.num_bits <= len(in_data):
                    # Extract the bits directly from the current chunk
                    payload_bits = in_data[tag_pos_window:tag_pos_window + self.num_bits]
                    self.process_payload(tag, payload_bits)
                else:
                    # Start buffering if data spans multiple chunks
                    self.buffering_active = True
                    self.pending_tag = tag  # Store the tag for later use
                    self.buffer = list(in_data[tag_pos_window:])  # Add available bits to buffer

                break  # Only analyse first triggering tag (in case there is more than one)

        return len(output_items[0])
    

    def print_log(self, tag, payload_and_CRC):
        print(f"Packet start sample: {tag.offset}, ")
        print(f"length: {self.num_bytes - self.CRC_size}, ")
        print("CRC: " + " ".join(f"{byte:02X}" for byte in payload_and_CRC[-self.CRC_size:]))
        print(" ".join(f"{byte:02X}" for byte in payload_and_CRC[:self.num_bytes - self.CRC_size]))
        print("")

    def process_payload(self, tag, payload_bits):
        # Convert bits to decimal
        payload_and_CRC = self.binary_to_uint8_list(payload_bits)
        self.print_log(tag, payload_and_CRC)
        
        self.add_item_tag(
            0,  # Output port
            tag.offset + self.num_bits,
            self.output_tag_key,  # String key
            gr.pmt.from_long(self.num_bytes),  # Value
        )

    def process_buffered_data(self):
        # Process the buffered data once it has enough bits
        payload_bits = self.buffer[:self.num_bits]
        self.process_payload(self.pending_tag, payload_bits)

        # Empty the buffer and reset state
        self.buffer = []
        self.buffering_active = False
        self.pending_tag = None
        self.num_bits = 0

    def binary_to_uint8_list(self, binary):
        # Ensure the binary list length is a multiple of 8
        if len(binary) % 8 != 0:
            raise ValueError("The binary list length must be a multiple of 8.")

        uint8_list = []
        for i in range(0, len(binary), 8):
            # Extract 8 bits
            byte = binary[i:i+8]

            # Convert LSB to MSB bits to integer
            uint8 = 0
            for i, bit in enumerate(byte):
                uint8 |= (bit << i)
            uint8_list.append(uint8)

        return uint8_list

