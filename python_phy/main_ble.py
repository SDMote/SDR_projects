import click
import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter, add_awgn
from visualisation import subplots_iq_spectrogra_bits, plot_payload
from receiver import ReceiverBLE


@click.command()
@click.option("--filename", default="BLE_0dBm.dat", type=str, help="The name of the data file to process.")
@click.option("--fs", default=10e6, type=float, help="Sampling frequency in Hz (default: 10e6).")
@click.option("--decimation", default=1, type=int, help="Decimation factor (default: 1).")
def main(filename: str, fs: float, decimation: int) -> None:
    """Process IQ data from file."""

    # For now assume BLE 1 Mb/s. TODO: consider BLE 2 Mb/s
    transmission_rate = 1e6  # BLE 1 Mb/s
    sps: float = fs / transmission_rate

    # Base address as defined in DotBot radio_default.h
    base_address: int = 0x12345678

    # Open file
    iq_samples = read_iq_data(f"../capture_nRF/data/new/{filename}")

    # Pre-processing
    # iq_samples = add_awgn(iq_samples, snr_db=4)

    # Low pass filter
    iq_samples = decimating_fir_filter(
        iq_samples, decimation=decimation, gain=1, fs=fs, cutoff_freq=1e6, transition_width=500e3, window="hamming"
    )

    # Squelch
    iq_samples = simple_squelch(iq_samples, threshold=10e-2)

    # Initialise the receiver and process data
    receiver = ReceiverBLE(fs=fs / decimation, sps=sps / decimation)
    bit_samples = receiver.demodulate(iq_samples)  # From IQ samples to hard decisions
    received_packets: list[dict] = receiver.process_phy_packet(
        bit_samples, base_address
    )  # From hard decisions to packets

    # Print results
    print(received_packets)

    # Plot
    subplots_iq_spectrogra_bits([iq_samples, bit_samples], fs=fs / decimation, show=False)
    if received_packets:
        plot_payload(received_packets[0])
    plt.show()


if __name__ == "__main__":
    main()
