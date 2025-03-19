import click
import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import add_white_gaussian_noise
from visualisation import subplots_iq_spectrogram_bits, plot_payload
from receiver import Receiver802154


@click.command()
@click.option("--filename", default="802154_0dBm.dat", type=str, help="The name of the data file to process.")
@click.option("--fs", default=10e6, type=float, help="Sampling frequency in Hz (default: 10e6).")
@click.option(
    "--crc_included",
    default=True,
    type=bool,
    help="IEEE 802.15.4 includes the CRC in the last two bytes of the payload, but it's not mandatory (default: True).",
)
@click.option(
    "--preamble_detection_threshold",
    default=12,
    type=int,
    help="How many bit errors are accepted when detecting the preamble (default: 12).",
)
def main(filename: str, fs: float, crc_included: bool, preamble_detection_threshold: int) -> None:
    """Process IQ data from file."""

    # Open file
    iq_samples = read_iq_data(f"../capture_nRF/data/new/{filename}")

    # Simulate channel
    # iq_samples = add_awgn(iq_samples, snr_db=4)

    # Initialise the receiver and process data
    receiver = Receiver802154(fs=fs)
    chip_samples = receiver.demodulate(iq_samples)  # From IQ samples to hard decisions
    received_packets: list[dict] = receiver.process_phy_packet(
        chip_samples, preamble_threshold=preamble_detection_threshold, CRC_included=crc_included
    )  # From hard decisions to packets

    # Print results
    print(received_packets)

    # Plot
    subplots_iq_spectrogram_bits([iq_samples, chip_samples], fs=fs, show=False)
    if received_packets:
        plot_payload(received_packets[0])
    plt.show()


if __name__ == "__main__":
    main()
