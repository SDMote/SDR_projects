import click
import matplotlib.pyplot as plt
from data_io import read_iq_data
from visualisation import plot_payload, subplots_iq, plot_ber_vs_frequency_offset
from receiver import ReceiverBLE, Receiver802154
from transmitter import TransmitterBLE, Transmitter802154
from interference_utils import (
    subtract_interference_wrapper,
    compute_ber_vs_frequency,
)
import numpy as np


@click.command()
@click.argument("interference", type=click.Choice(["ble", "802154"]))
@click.argument("affected", type=click.Choice(["ble", "802154"]))
def main(affected, interference):
    """
    Subtract known interference synthesising higher power signal after demodulating it.

    Affected must be either 'ble' or '802154'.
    Interference can be 'ble', '802154'.
    """
    if affected == interference:
        click.echo("Error: 'interference' must be different from 'affected'.")
        return

    # Hardcoded filenames for now
    if affected == "ble":
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
    if affected == "ble":
        affected_receiver = ReceiverBLE(fs=fs)
    else:
        affected_receiver = Receiver802154(fs=fs)

    """Open affected and interference files"""
    iq_affected = read_iq_data(f"{relative_path}{affected_filename}")

    """This is different from the known interference scenario, the interference is demodulated first and then modulated."""
    if interference == "ble":
        interference_receiver = ReceiverBLE(fs=fs)
        interference_transmitter = TransmitterBLE(fs=fs)
    else:
        interference_receiver = Receiver802154(fs=fs)
        interference_transmitter = Transmitter802154(fs=fs)

    # Demodulate the interference
    bit_samples = interference_receiver.demodulate(iq_affected)  # From IQ samples to hard decisions
    interference_packet: list[dict] = interference_receiver.process_phy_packet(bit_samples)
    interference_packet: dict = interference_packet[0]
    print(interference_packet["payload"])

    # Synthesise the interference
    iq_interference = interference_transmitter.process_phy_payload(interference_packet["payload"])
    iq_interference = interference_transmitter.modulate(iq_interference, zero_padding=500)

    iq_interference_known = read_iq_data(f"{relative_path}{interference_filename}")
    # iq_interference = iq_interference_known

    """Subtract the synthesised interference and demodulate (blind analysis)"""
    freq_range = range(0, 35000, 50)
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

    # """BER analysis with frequency variations (non-blind analysis)"""
    # bit_error_rates = compute_ber_vs_frequency(
    #     freq_range,
    #     affected=iq_affected,
    #     interference=iq_interference,
    #     fs=fs,
    #     reference_packet=reference_packet,
    #     receiver=affected_receiver,
    # )
    # plot_ber_vs_frequency_offset(freq_range, bit_error_rates, show=False)
    plt.show()


if __name__ == "__main__":
    main()
