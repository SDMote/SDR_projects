import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter, add_awgn
from visualisation import create_subplots
from receiver import ReceiverBLE
from packet_utils import correlate_access_code, map_nibbles_to_chip_string

CHIP_MAPPING: np.ndarray = np.array(
    [
        0xE077AE6C,  # 0
        0xCE077AE6,  # 1
        0x6CE077AE,  # 2
        0xE6CE077A,  # 3
        0xAE6CE077,  # 4
        0x7AE6CE07,  # 5
        0x77AE6CE0,  # 6
        0x877AE6CE,  # 7
        0x1F885193,  # 8
        0x31F88519,  # 9
        0x931F8851,  # A
        0x1931F885,  # B
        0x51931F88,  # C
        0x851931F8,  # D
        0x8851931F,  # E
        0x78851931,  # F
    ],
    dtype=np.uint32,
)


if __name__ == "__main__":
    filename: str = "802154_0dBm.dat"
    fs: int | float = 10e6  # Hz
    sps: int | float = 5
    decimation: int = 1

    # Open file
    iq_samples = read_iq_data(f"../capture_nRF/data/{filename}")

    # Pre-processing
    # iq_samples = add_awgn(iq_samples, snr_db=4)

    # Low pass filter
    iq_samples = decimating_fir_filter(
        iq_samples, decimation=decimation, gain=1, fs=fs, cutoff_freq=3e6, transition_width=1000e3, window="hamming"
    )

    # Squelch
    iq_samples = simple_squelch(iq_samples, threshold=10e-2)

    # Initialise the receiver and process data
    receiver = ReceiverBLE(fs=fs / decimation, sps=sps / decimation)
    bit_samples = receiver.demodulate(iq_samples)  # From IQ samples to hard decisions
    string_access_code = map_nibbles_to_chip_string([0x00, 0x00, 0x00, 0x00, 0xA7], CHIP_MAPPING)
    print(f"{string_access_code = }")
    preamble_positions = correlate_access_code(bit_samples, string_access_code, 0, reduce_mask=True)
    print(f"{preamble_positions = }")

    # received_packets: dict = receiver.process_phy_packet(bit_samples, base_address)  # From hard decisions to packets

    # Print results
    # print(received_packets)

    # Plot
    create_subplots([iq_samples, bit_samples], fs=fs / decimation)
