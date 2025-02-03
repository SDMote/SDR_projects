"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class ReadPayloadBLE(gr.sync_block):
    def __init__(self, input_tag="payload start"):
        gr.sync_block.__init__(
            self,
            name="Read Payload BLE",
            in_sig=[np.uint8],  # Input is binary stream
            out_sig=[np.uint8],  # Pass-through output stream
        )
        self.input_tag_key = gr.pmt.intern(input_tag)  # Tag to search for
        self.CRC_size = 3  # CRC size in bytes

        self.total_bits = None  # Number of bits to read
        self.buffer = np.array([], dtype=np.uint8)  # Buffer to hold data across chunks
        self.buffering_active = False  # Flag to indicate active buffering
        self.pending_tag = None  # Store the tag for processing when buffer is filled

        self.lfsr = None

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

                    # Initialise payload reading parameters
                    self.S0 = (gr.pmt.u8vector_elements(tag.value)[0]) # S0 value, necessary for CRC calculation
                    self.total_bytes = (gr.pmt.u8vector_elements(tag.value)[1] + self.CRC_size)  # Number of bytes to read
                    self.total_bits = int(8 * self.total_bytes)  # Number of bits to read
                    self.lfsr = gr.pmt.u8vector_elements(tag.value)[2]

                    # Check if the required number of bits is within the current chunk
                    if tag_pos_window + self.total_bits <= len(in_data):
                        # Extract the bits directly from the current chunk
                        payload_bits = in_data[tag_pos_window:tag_pos_window + self.total_bits]
                        self.process_payload(tag, payload_bits)
                    else:
                        # Start buffering if data spans multiple chunks
                        self.buffering_active = True
                        self.pending_tag = tag  # Store the tag for later use
                        self.buffer = np.array(in_data[tag_pos_window:], dtype=np.uint8)  # Add available bits to buffer

                    break  # Only analyse first triggering tag (in case there is more than one in the same window)

        return len(output_items[0])
    

    def print_log(self, tag, payload_and_CRC, CRC_check):
        print(f"Packet start sample: {tag.offset}, ")
        print(f"length: {self.total_bytes - self.CRC_size}, ")
        crc_str = " ".join(f"{byte:02X}" for byte in payload_and_CRC[-self.CRC_size:])
        print(f"CRC: {crc_str}, {'OK' if CRC_check else 'ERROR'}")
        print(" ".join(f"{byte:02X}" for byte in payload_and_CRC[:self.total_bytes - self.CRC_size]))
        print("")

    def process_payload(self, tag, payload_bits):
        payload_and_CRC = self.binary_to_uint8_array(payload_bits)  # Convert bits to decimal
        payload_and_CRC = self.BLE_whitening(payload_and_CRC, lfsr=self.lfsr)  # Dewhiten data

        # CRC check
        header = np.array([self.S0, self.total_bytes - self.CRC_size], dtype=np.uint8)
        header_and_payload = np.concatenate((header, payload_and_CRC[:-self.CRC_size]))
        computed_CRC = self.compute_CRC(header_and_payload)
        CRC_check = True if (computed_CRC == payload_and_CRC[-self.CRC_size:]).all() else False

        # Print log (use this section for storing the data as well)
        self.print_log(tag, payload_and_CRC, CRC_check)

        # Tagging is bugged so no tags are added

    def process_buffered_data(self):
        # Process the buffered data once it has enough bits
        payload_bits = self.buffer[:self.total_bits]
        self.process_payload(self.pending_tag, payload_bits)

        # Reset buffering state
        self.buffering_active = False
        self.pending_tag = None
        self.total_bits = None
        self.total_bytes = None
        self.lfsr = None
        self.S0 = None
    
    def binary_to_uint8_array(self, binary):       
        # Ensure the binary array length is a multiple of 8
        if len(binary) % 8 != 0:
            raise ValueError(f"The binary list {len(binary)} length must be a multiple of 8.")
        
        # Convert binary to NumPy array
        binary_array = np.array(binary, dtype=np.uint8)
        binary_array = binary_array.reshape(-1, 8)[:, ::-1] #  LSB to MSB correction
        uint8_array = np.packbits(binary_array, axis=1).flatten()

        return uint8_array
    
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
                if lfsr_msb:  # If MSB is 1 (before shifting), apply feedback
                    lfsr = (lfsr << 1) ^ polynomial  # XOR with predefined polynomial
                else:
                    lfsr <<= 1
                lfsr &= 0x7F  # 0x7F mask to keep ksfr within 7 bits

            output[idx] = whitened_byte

        return output
    
    def compute_CRC(self, data, crc_init=0x00FFFF, crc_poly=0x00065B, crc_size=3):
        crc_mask = (1 << (crc_size * 8)) - 1  # Mask to n-byte width (0xFFFF for crc_size = 2)
        
        def swap_nbit(num, n):
            num = num & crc_mask
            reversed_bits = f'{{:0{n * 8}b}}'.format(num)[::-1]
            return int(reversed_bits, 2)
        
        crc_init = swap_nbit(crc_init, crc_size)  # LSB -> MSB
        crc_poly = swap_nbit(crc_poly, crc_size)
        
        crc = crc_init
        for byte in data:
            crc ^= int(byte)
            for _ in range(8):  # Process each bit
                if crc & 0x01:  # Check the LSB
                    crc = (crc >> 1) ^ crc_poly
                else:
                    crc >>= 1
                crc &= crc_mask  # Ensure CRC size
        
        return np.array([(crc >> (8 * i)) & 0xFF for i in range(crc_size)], dtype=np.uint8)

