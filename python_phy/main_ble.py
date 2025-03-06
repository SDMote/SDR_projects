import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter, add_awgn
from visualisation import create_subplots, plot_payload
from receiver import ReceiverBLE


if __name__ == "__main__":
    filename: str = "BLE_tone_0dBm_8dBm_0MHz.dat"
    fs: int | float = 10e6  # Hz
    sps = 10
    decimation: int = 1
    base_address = 0x12345678  # As defined in DotBot radio_default.h

    # Open file
    iq_samples = read_iq_data(f"../capture_nRF/data/new/{filename}")

    # Pre-processing
    # iq_samples = add_awgn(iq_samples, snr_db=4)

    # Low pass filter
    iq_samples = decimating_fir_filter(
        iq_samples, decimation=decimation, gain=1, fs=fs, cutoff_freq=1.5e6, transition_width=1000e3, window="hamming"
    )

    # Squelch
    iq_samples = simple_squelch(iq_samples, threshold=10e-2)

    # Initialise the receiver and process data
    receiver = ReceiverBLE(fs=fs / decimation, sps=sps / decimation)
    bit_samples = receiver.demodulate(iq_samples)  # From IQ samples to hard decisions
    received_packets: dict = receiver.process_phy_packet(bit_samples, base_address)  # From hard decisions to packets

    # Print results
    print(received_packets)

    # Plot
    create_subplots([iq_samples, bit_samples], fs=fs / decimation, show=False)
    plot_payload(received_packets[0])
    plt.show()
