import click
import numpy as np
from sic_simulator import SimulationConfig, SimulatorSIC
from data_io import sic_save_simulation


@click.command()
@click.option(
    "--protocol-high",
    required=True,
    type=click.Choice(["ble", "802154"]),
    help="High-power signal protocol (BLE or 802154).",
)
@click.option(
    "--protocol-low",
    required=True,
    type=click.Choice(["ble", "802154"]),
    help="Low-power signal protocol (BLE or 802154).",
)
@click.option("--ble-rate", default=1e6, type=float, help="BLE data rate (1e6 or 2e6).")
@click.option("--payload-len-high", default=30, type=int, help="Bytes in high-power payload.")
@click.option("--payload-len-low", default=200, type=int, help="Bytes in low-power payload.")
@click.option("--num-trials", default=4, type=int, help="Number of Monte Carlo trials.")
@click.option("--sampling-rate", default=10e6, type=float, help="Sampling rate in samples/second")
def run_simulation(protocol_high, protocol_low, ble_rate, payload_len_high, payload_len_low, num_trials, sampling_rate):
    cfg = SimulationConfig(
        sampling_rate=sampling_rate,  # Samples per second
        protocol_high=protocol_high,  # BLE or IEEE 802.15.4
        protocol_low=protocol_low,
        ble_rate=ble_rate,  # 1 Mb/s or 2 Mb/s
        amplitude_high=10 ** (-6 / 20),  # Amplitude for higher-power signal
        amplitude_low=10 ** (-7 / 20),  # Amplitude for lower-power signal
        snr_low_db=0,  # SNR in (dB) relative to the lower power generated signal
        freq_offset_range=range(-5000, 5000, 50),  # (Hz) For demodulation brute force search
        fine_step=2,  # Step size (Hz) for the fine search
        fine_window=50,  # Half-width (Hz) of the window around best coarse frequency
        payload_len_high=payload_len_high,  # Bytes in high-power payload
        payload_len_low=payload_len_low,  # Bytes in low-power payload
        sample_shift_range_low=(0, 1),  # Range to randomly apply fractional delays on IQ data
        sample_shift_range_high=(200, 2200),
        freq_high=None,  # Fixed frequency offset for high-power signal, if None random in freq_offset_range
        freq_low=None,  # Fixed frequency offset for low-power signal, if None random in freq_offset_range
        phase_high=None,  # Fixed phase for high-power signal, if None random
        phase_low=None,  # Fixed phase for low-power signal, if None random
        adc_bits=12,  # ADC resolution in bits
        adc_vmax=1.0,  # Maximum ADC input amplitude
        padding=500,  # Zero-pad the generated signals before adding them to ensure equal length
    )

    high_power_db = -6  # dB, around 0.707 amplitude
    low_powers_db = np.arange(-6, -20, -1)  # dB, up to around 0.1 amplitude
    snr_lows_db = np.arange(0, 15, 2)  # dB

    simulator = SimulatorSIC(cfg)
    pdr: np.ndarray = simulator.run_monte_carlo_parallel(
        high_power_db, low_powers_db, snr_lows_db, num_trials=num_trials
    )

    sic_save_simulation(cfg, high_power_db, low_powers_db, snr_lows_db, num_trials, pdr, folder="./sic_simulations")


if __name__ == "__main__":
    run_simulation()
