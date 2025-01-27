"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class tag_payload_start(gr.sync_block):
    def __init__(self, num_bits=8):
        gr.sync_block.__init__(
            self,
            name="Tag Payload Start",
            in_sig=[np.uint8],  # Input is binary stream
            out_sig=[np.uint8],  # Pass-through output stream
        )
        self.tag_key = gr.pmt.intern("length start")  # Tag to search for
        self.num_bits = num_bits  # Number of bits to read (parameterised)
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

        # Handle active buffering if needed
        if self.buffering_active:
            self.buffer.extend(in_data)
            if len(self.buffer) >= self.num_bits:  # Check if we have enough bits
                self.process_buffered_data()
                
        else:      
            # If not buffering, look for a triggering tag that starts buffering or direct processing if possible within the window
            for tag in tags:
                if tag.key == self.tag_key:  # Look for triggering tag
                    tag_pos_window = tag.offset - nread  # Position of the tag in the current window
                    
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

    def process_payload(self, tag, payload_bits):
        # Convert bits to decimal
        payload_length = self.binary_to_decimal(payload_bits)

        # Add a new tag for the start of the payload
        self.add_item_tag(
            0,  # Output port
            tag.offset + self.num_bits,  # Absolute position of the payload start
            gr.pmt.intern("payload start"),
            gr.pmt.from_long(payload_length),  # Length of the payload
        )

    def process_buffered_data(self):
        # Process the buffered data once it has enough bits
        payload_bits = self.buffer[:self.num_bits]
        self.process_payload(self.pending_tag, payload_bits)

        # Empty the buffer and reset state
        self.buffer = []
        self.buffering_active = False
        self.pending_tag = None

    def binary_to_decimal(self, binary):
        # Convert LSB to MSB bits to integer
        decimal = 0
        for i, bit in enumerate(binary):
            decimal |= (bit << i)

        return decimal
