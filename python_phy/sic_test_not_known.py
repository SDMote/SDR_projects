import click
import matplotlib.pyplot as plt
from data_io import read_iq_data
from visualisation import plot_payload, subplots_iq
from receiver import ReceiverBLE, Receiver802154
from transmitter import TransmitterBLE, Transmitter802154
from interference_utils import subtract_interference_wrapper
import numpy as np


def successive_interference_cancellation(
    mixed_signal: np.ndarray,
    receiver_high: object,  # Receiver for the stronger signal
    transmitter_high: object,  # Transmitter for the stronger signal
    receiver_low: object,  # Receiver for the weaker signal
    sample_rate: float,
    freq_range: range,
    verbose: bool = True,
) -> tuple:
    """
    Perform successive interference cancellation:
    1. Demodulate stronger signal
    2. Synthesise stronger signal
    3. Subtract stronger signal from mixed signal
    4. Demodulate weaker signal from residual
    """
    # Demodulate interference
    bit_samples = receiver_high.demodulate(mixed_signal)
    interference_packets = receiver_high.process_phy_packet(bit_samples)

    if not interference_packets:
        raise ValueError("No interference packets detected")

    interference_packet = interference_packets[0]
    if verbose:
        print(interference_packet["payload"])

    # Synthesize interference
    iq_baseband = transmitter_high.process_phy_payload(interference_packet["payload"])
    synthesized_interference = transmitter_high.modulate(iq_baseband, zero_padding=500)

    # Subtract interference
    subtracted_signal = subtract_interference_wrapper(
        affected=mixed_signal,
        interference=synthesized_interference,
        fs=sample_rate,
        freq_offsets=freq_range,
        verbose=verbose,
    )

    # Demodulate affected signal
    bit_samples = receiver_low.demodulate(subtracted_signal)
    affected_packets = receiver_low.process_phy_packet(bit_samples)

    if not affected_packets:
        raise ValueError("No affected packets detected after cancellation")

    return affected_packets[0], subtracted_signal, synthesized_interference


@click.command()
@click.argument("interference", type=click.Choice(["ble", "802154"]))
@click.argument("affected", type=click.Choice(["ble", "802154"]))
def main(affected, interference):

    # Hardcoded filenames for now
    if affected == "ble":
        if interference == "802154":
            affected_filename = "BLE_802154_0dBm_8dBm_0MHz.dat"
        elif interference == "tone":
            affected_filename = "BLE_tone_0dBm_8dBm_0MHz.dat"
        else:
            click.echo("Unsupported interference for affected 'ble'.")
            return
    elif affected == "802154":
        if interference == "ble":
            affected_filename = "802154_BLE_0dBm_8dBm_0MHz.dat"
        elif interference == "tone":
            affected_filename = "802154_tone_0dBm_8dBm_0MHz.dat"
        else:
            click.echo("Unsupported interference for affected '802154'.")
            return
    else:
        click.echo("Unsupported value for 'affected'.")
        return

    # Hardcoded parameters
    sample_rate = 10e6
    relative_path = "../capture_nRF/data/new/"
    freq_range = range(0, 35000, 50)

    # Initialise receivers/transmitters
    if affected == "ble":
        affected_receiver = ReceiverBLE(fs=sample_rate)
    else:
        affected_receiver = Receiver802154(fs=sample_rate)

    if interference == "ble":
        interference_receiver = ReceiverBLE(fs=sample_rate)
        interference_transmitter = TransmitterBLE(fs=sample_rate)
    else:
        interference_receiver = Receiver802154(fs=sample_rate)
        interference_transmitter = Transmitter802154(fs=sample_rate)

    iq_affected = read_iq_data(f"{relative_path}{affected_filename}")

    try:
        affected_packet, subtracted, synthesized_interference = successive_interference_cancellation(
            mixed_signal=iq_affected,
            receiver_high=interference_receiver,
            transmitter_high=interference_transmitter,
            receiver_low=affected_receiver,
            sample_rate=sample_rate,
            freq_range=freq_range,
            verbose=True,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    # Plot results
    subplots_iq(
        [iq_affected, synthesized_interference, subtracted],
        sample_rate,
        ["Interfered packet", "Synthesized Interference", "Subtracted"],
        show=False,
    )
    plot_payload(affected_packet)

    plt.show()


if __name__ == "__main__":
    main()
