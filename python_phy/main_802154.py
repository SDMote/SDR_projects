import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter, add_awgn
from visualisation import create_subplots
from receiver import Receiver802154


if __name__ == "__main__":
    filename: str = "802154_0dBm.dat"
    fs: int | float = 10e6  # Hz
    sps: int | float = 5
    decimation: int = 1
    preamble_detection_threshold: int = 12
    CRC_included: bool = True

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
    receiver = Receiver802154(fs=fs / decimation, sps=sps / decimation)
    chip_samples = receiver.demodulate(iq_samples)  # From IQ samples to hard decisions
    received_packets: dict = receiver.process_phy_packet(
        chip_samples, preamble_threshold=preamble_detection_threshold, CRC_included=CRC_included
    )  # From hard decisions to packets

    # Print results
    print(received_packets)

    # Plot
    create_subplots([iq_samples, chip_samples], fs=fs / decimation)
