"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class tag_payload_start(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="Tag Payload Start",
            in_sig=[np.uint8],  # Input is binary stream
            out_sig=[np.uint8],  # Pass-through output stream
        )

    def work(self, input_items, output_items):
        in_data = input_items[0]  # Input stream
        out_data = output_items[0]  # Output stream
        nread = self.nitems_read(0)  # Number of items read so far on input
        tags = self.get_tags_in_window(0, 0, len(in_data))  # Tags in current window

        # Copy input to output (pass-through block)
        out_data[:] = in_data

        for tag in tags:
            if tag.key == gr.pmt.intern("length start"):  # Look for "length start" tag
                tag_pos_window = tag.offset - nread  # Position of the tag in the current window
                
                # Ensure there is data to extract the length byte
                if tag_pos_window + 8 <= len(in_data):
                    # Extract 8 bits following the tag               
                    payload_length_bits = in_data[tag_pos_window:tag_pos_window + 8] 

                    # Convert LSB to MSB 8 bits to integer
                    payload_length = 0
                    for i, bit in enumerate(payload_length_bits):
                        payload_length |= (bit << i)

                    # Add a new tag for the start of the payload
                    self.add_item_tag(
                        0,  # Output port
                        tag.offset + 8,  # Absolute position of the payload start
                        gr.pmt.intern(f"payload start"),
                        gr.pmt.from_long(payload_length),  # Length of the payload
                    )

        return len(in_data)
