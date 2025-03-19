import click
import numpy as np
import matplotlib.pyplot as plt
import scipy
from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter
from visualisation import plot_payload, compare_bits_with_reference, subplots_iq, plot_ber_vs_frequency_offset
from receiver import ReceiverBLE, Receiver802154
from interference_utils import (
    multiply_by_complex_exponential,
    pad_interference,
    correlation_wrapper,
    subtract_interference_wrapper,
    compute_ber_vs_frequency,
)


@click.command()
@click.argument("affected", type=click.Choice(["ble", "802154"]))
@click.argument("interference", type=click.Choice(["ble", "802154", "tone"]))
def main(affected, interference):
    """
    Subtract known interference using offline measurements.

    Affected must be either 'ble' or '802154'.
    Interference can be 'ble', '802154' or 'tone'.
    """
    if affected == interference:
        click.echo("Error: 'interference' must be different from 'affected'.")
        return

    # Hardcoded filenames for now
    if affected == "ble":
        reference_filename = "BLE_0dBm.dat"
        if interference == "802154":
            affected_filename = "BLE_802154_0dBm_8dBm_0MHz.dat"
            interference_filename = "802154_8dBm_0MHz.dat"
        elif interference == "tone":
            affected_filename = "BLE_tone_0dBm_8dBm_0MHz.dat"
            interference_filename = "tone_8dBm_0MHz_BLE.dat"
        else:
            click.echo("Unsupported interference for affected 'ble'.")
            return
    elif affected == "802154":
        reference_filename = "802154_0dBm.dat"
        if interference == "ble":
            affected_filename = "802154_BLE_0dBm_8dBm_0MHz.dat"
            interference_filename = "BLE_8dBm_0MHz.dat"
        elif interference == "tone":
            affected_filename = "802154_tone_0dBm_8dBm_0MHz.dat"
            interference_filename = "tone_8dBm_0MHz_802154.dat"
        else:
            click.echo("Unsupported interference for affected '802154'.")
            return
    else:
        click.echo("Unsupported value for 'affected'.")
        return

    """Open reference packet (for BER comparison)"""
    # Hardcoded parameters for now
    fs: int | float = 10e6  # Hz
    relative_path: str = "../capture_nRF/data/new/"

    # Open file, demodulate and get reference packet
    iq_reference = read_iq_data(f"{relative_path}{reference_filename}")
    if affected == "ble":
        affected_receiver = ReceiverBLE(fs=fs)
    else:
        affected_receiver = Receiver802154(fs=fs)

    bit_samples = affected_receiver.demodulate(iq_reference)  # From IQ samples to hard decisions
    reference_packet: list[dict] = affected_receiver.process_phy_packet(bit_samples)  # From hard decisions to packets
    reference_packet: dict = reference_packet[0]

    """Open affected and interference files"""
    iq_affected = read_iq_data(f"{relative_path}{affected_filename}")
    iq_interference = read_iq_data(f"{relative_path}{interference_filename}")

    """Subtract the known interference and demodulate (blind analysis)"""
    freq_range = range(0, 12000, 20)
    subtracted = subtract_interference_wrapper(
        affected=iq_affected, interference=iq_interference, fs=fs, freq_offsets=freq_range, verbose=True
    )
    subplots_iq(
        [iq_affected, iq_interference, subtracted], fs, ["Interfered packet", "Interference", "Subtracted"], show=False
    )

    # Initialise the receiver and process data
    bit_samples = affected_receiver.demodulate(subtracted)  # From IQ samples to hard decisions
    reference_packet: list[dict] = affected_receiver.process_phy_packet(bit_samples)  # From hard decisions to packets
    if reference_packet:
        reference_packet: dict = reference_packet[0]
        plot_payload(reference_packet)

    """BER analysis with frequency variations (non-blind analysis)"""
    bit_error_rates = compute_ber_vs_frequency(
        freq_range,
        affected=iq_affected,
        interference=iq_interference,
        fs=fs,
        reference_packet=reference_packet,
        receiver=affected_receiver,
    )
    plot_ber_vs_frequency_offset(freq_range, bit_error_rates, show=False)
    plt.show()


if __name__ == "__main__":
    main()
