from dataclasses import dataclass
import numpy as np
from typing import Type
import matplotlib.pyplot as plt

from receiver import Receiver, ReceiverType, ReceiverBLE, Receiver802154, adc_quantise
from transmitter import Transmitter, TransmitterBLE, Transmitter802154
from interference_utils import multiply_by_complex_exponential, subtract_interference_wrapper
from snr_related import add_white_gaussian_noise
from filters import fractional_delay_fir_filter
from visualisation import subplots_iq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class SimulationConfig:
    """Successive Interference Cancellation simulation parameters"""

    sampling_rate: float  # Samples per second
    protocol_high: ReceiverType  # BLE or IEEE 802.15.4
    protocol_low: ReceiverType
    ble_rate: float  # 1 Mb/s or 2 Mb/s
    amplitude_high: float  # Amplitude for higher-power signal (fixed at ADC vmax)
    amplitude_low: float  # Amplitude for lower-power signal (swept)
    snr_low_db: float  # SNR in (dB) relative to the lower power generated signal
    freq_offset_range: range  # (Hz) For demodulation brute force search
    fine_step: float | None  # Step size (Hz) for the fine search
    fine_window: float | None  # Half-width (Hz) of the window around best coarse frequency
    payload_len_high: int  # Bytes in high-power payload
    payload_len_low: int  # Bytes in low-power payload
    sample_shift_range_low: tuple[int, int]  # Range to randomly apply fractional delays on IQ data
    sample_shift_range_high: tuple[int, int]
    freq_high: float = None  # Fixed frequency offset for high-power signal, if None random in freq_offset_range
    freq_low: float = None  # Fixed frequency offset for low-power signal, if None random in freq_offset_range
    phase_high: float = None  # Fixed phase for high-power signal, if None random
    phase_low: float = None  # Fixed phase for low-power signal, if None random
    adc_bits: int = 12  # ADC resolution in bits
    adc_vmax: float = 1.0  # Maximum ADC input amplitude
    padding: int = 500  # Zero-pad the generated signals before adding them to ensure equal length


class SimulatorSIC:
    def __init__(self, config: SimulationConfig) -> None:
        """
        Wrapper class to simulate PDR against variable power differences.
        """
        self.cfg = config

        # Map protocol names to their (Tx class, Rx class, default rate)
        proto_map: dict[
            str,
            tuple[
                Type[Transmitter],
                Type[Receiver],
                float,  # Default BLE rate, ignored for non‐BLE
            ],
        ] = {
            "ble": (TransmitterBLE, ReceiverBLE, config.ble_rate),
            "802154": (Transmitter802154, Receiver802154, 0.0),  # Rate unused
        }

        # Helper to instantiate Tx/Rx with or without rate param
        def _map_protocol(proto: str):
            TxClass, RxClass, rate = proto_map[proto]
            if proto == "ble":  # Pass both sampling_rate and transmission_rate

                return (
                    TxClass(self.cfg.sampling_rate, transmission_rate=rate),
                    RxClass(self.cfg.sampling_rate, transmission_rate=rate),
                )
            else:  # Only pass sampling_rate
                return (
                    TxClass(self.cfg.sampling_rate),
                    RxClass(self.cfg.sampling_rate),
                )

        self.transmitter_high, self.receiver_high = _map_protocol(self.cfg.protocol_high)
        self.transmitter_low, self.receiver_low = _map_protocol(self.cfg.protocol_low)

    # Wrapper to generate baseband IQ samples for a single transmitter with given parameters.
    def _generate_signal(
        self,
        payload: np.ndarray,
        transmitter: Transmitter,
        amplitude: float,
        freq_offset: float,
        phase: float,
        zero_padding: int,
    ) -> np.ndarray:
        """
        Wrapper to generate baseband IQ samples for a single transmitter with given parameters.
        """
        iq = transmitter.modulate_from_payload(payload, zero_padding=zero_padding)  # Unit amplitude baseband
        freq_offset = (
            freq_offset
            if freq_offset is not None
            else np.random.uniform(self.cfg.freq_offset_range.start, self.cfg.freq_offset_range.stop)
        )
        phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        return multiply_by_complex_exponential(iq, self.cfg.sampling_rate, freq_offset, phase, amplitude)

    # Helper function to equalise the size of two arrays
    def _zero_padding(self, array1: np.ndarray, array2: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray]:
        target_length = max(len(array1), len(array2)) + padding

        # Add left and right padding to match target length
        array1 = np.pad(array1, (padding, target_length - len(array1)))
        array2 = np.pad(array2, (padding, target_length - len(array2)))

        return array1, array2

    # Run a single trial of SIC. Returns tuple: (delivery_success_high, delivery_success_low)
    def simulate_single_trial(
        self, amplitude_high: float, amplitude_low: float, snr_low_db: float = None, *, verbose: bool = False
    ) -> tuple[bool, bool]:
        """
        Run a single trial of SIC. Returns tuple: (delivery_success_high, delivery_success_low)
        """
        # Generate payload outside _generate_signal() method in case we want to compute BER in the future
        payload_high = np.random.randint(0, 256, size=self.cfg.payload_len_high, dtype=np.uint8)
        payload_low = np.random.randint(0, 256, size=self.cfg.payload_len_low, dtype=np.uint8)

        # Generate IQ signals independently
        zero_padding: int = 10 + int(
            np.ceil(np.max(self.cfg.sample_shift_range_low + self.cfg.sample_shift_range_high))
        )  # Ensures no information loss/cropping
        iq_high = self._generate_signal(
            payload_high,
            self.transmitter_high,
            amplitude_high,
            self.cfg.freq_high,
            self.cfg.phase_high,
            zero_padding,
        )
        iq_low = self._generate_signal(
            payload_low,
            self.transmitter_low,
            amplitude_low,
            self.cfg.freq_low,
            self.cfg.phase_low,
            zero_padding,
        )

        # Mix and add noise: s1(t - shift1)A1·e^j(2pi·f1·t + phi1) + s2(t - shift2)A2·e^j(2pi·f2·t + phi2) + noise(t)
        # Add a random fractional delay fo both signals within a range set as config parameter
        iq_high = fractional_delay_fir_filter(iq_high, np.random.uniform(*self.cfg.sample_shift_range_high))
        iq_low = fractional_delay_fir_filter(iq_low, np.random.uniform(*self.cfg.sample_shift_range_low))
        iq_high, iq_low = self._zero_padding(iq_high, iq_low, padding=self.cfg.padding)  # Ensure same signals length

        # O-QPSK and FSK modulated signals' power is their amplitude squared
        noise_power = self.cfg.amplitude_low**2 / (10 ** (snr_low_db / 10))
        rx_iq: np.ndarray = add_white_gaussian_noise(iq_high + iq_low, noise_power, noise_power_db=False)

        rx_iq = adc_quantise(rx_iq, self.cfg.adc_vmax, self.cfg.adc_bits)  # Simulate ADC quantisation

        # Demodulate high-power signal first
        try:
            received_packet_high: list[dict] = self.receiver_high.demodulate_to_packet(rx_iq)
        except Exception:
            # Treat any exception as no packet received successfully
            received_packet_high: list[dict] = []

        if received_packet_high:
            received_packet_high: dict = received_packet_high[0]
            if verbose:
                print(f"{received_packet_high = }")
            success_high = received_packet_high["crc_check"]

            # Synthesise the high-power signal and subtract, even with wrong CRC check
            synth_high_iq = self.transmitter_high.modulate_from_payload(received_packet_high["payload"])
            subtracted_iq = subtract_interference_wrapper(
                rx_iq,
                synth_high_iq,
                self.cfg.sampling_rate,
                self.cfg.freq_offset_range,
                fine_step=self.cfg.fine_step,
                fine_window=self.cfg.fine_window,
                verbose=verbose,
            )

            if verbose:
                subplots_iq(
                    [rx_iq, synth_high_iq, subtracted_iq],
                    self.cfg.sampling_rate,
                    ["Interfered packet", "Synthesised", "Subtracted"],
                    show=True,
                )
        else:
            success_high = False
            # Cannot subtract interference, but still try to demodulate low-power signal
            subtracted_iq = rx_iq
            if verbose:
                print("High-power signal not demodulated; trying low-power directly")

        # Demodulate low-power signal
        try:
            received_packet_low: list[dict] = self.receiver_low.demodulate_to_packet(subtracted_iq)
        except Exception:
            # Treat any exception as no packet received successfully
            received_packet_low: list[dict] = []

        if received_packet_low:
            received_packet_low: dict = received_packet_low[0]
            if verbose:
                print(f"{received_packet_low = }")
            success_low = received_packet_low["crc_check"]
        else:
            return success_high, False

        return success_high, success_low

    # Sweep over the power and SNR of the low-power signal, and compute the PDR for both the high-power and low-power signals.
    def run_monte_carlo(
        self,
        high_power_db: float,
        low_powers_db: np.ndarray,  # Shape (Powers,)
        snr_lows_db: np.ndarray,  # Shape (SNRs,)
        *,
        num_trials: int,
    ) -> np.ndarray:  # Shape (2, Powers, SNRs)
        """
        Sweep over the power and SNR of the low-power signal and compute the PDR for both the high- and low-power signals.

        Returns
        -------
        pdr : np.ndarray
            Shape (2, Powers, SNRs), where
            - Axis 0 (Signal): 0 = high-power signal, 1 = low-power signal
            - Axis 1 (Powers): swept values for power differences
            - Axis 2 (SNRs): SNR values relative to the low-power signal
            Each pdr estimation is (num_successes / num_trials).
        """
        Powers = low_powers_db.size
        SNRs = snr_lows_db.size
        pdr = np.empty((2, Powers, SNRs), dtype=float)

        # Power dB -> amplitude
        high_amplitude: float = 10 ** (high_power_db / 20)
        low_amplitudes: np.ndarray = 10 ** (low_powers_db / 20)

        # Progress bar
        total_iterations = len(low_amplitudes) * len(snr_lows_db)
        progress_bar = tqdm(total=total_iterations, desc="Simulating", mininterval=5.0)

        for idx_power, amp_low in enumerate(low_amplitudes):
            for idx_snr, snr_low in enumerate(snr_lows_db):
                pdr_high, pdr_low = _worker_task(self, high_amplitude, amp_low, snr_low, num_trials)
                pdr[0, idx_power, idx_snr] = pdr_high
                pdr[1, idx_power, idx_snr] = pdr_low

                progress_bar.update(1)

        progress_bar.close()
        return pdr

    def run_monte_carlo_parallel(
        self,
        high_power_db: float,
        low_powers_db: np.ndarray,  # Shape (Powers,)
        snr_lows_db: np.ndarray,  # Shape (SNRs,)
        *,
        num_trials: int,
        max_workers: int = None,  # defaults to number of CPUs
    ) -> np.ndarray:  # Shape (2, Powers, SNRs)
        """
        Sweep over the power and SNR of the low-power signal and compute the PDR for both the high- and low-power signals.

        Returns
        -------
        pdr : np.ndarray
            Shape (2, Powers, SNRs), where
            - Axis 0 (Signal): 0 = high-power signal, 1 = low-power signal
            - Axis 1 (Powers): swept values for power differences
            - Axis 2 (SNRs): SNR values relative to the low-power signal
            Each pdr estimation is (num_successes / num_trials).
        """
        Powers: int = low_powers_db.size
        SNRs: int = snr_lows_db.size
        pdr: np.ndarray = np.empty((2, Powers, SNRs), dtype=float)

        # Power dB -> amplitude
        high_amplitude: float = 10 ** (high_power_db / 20)
        low_amplitudes: np.ndarray = 10 ** (low_powers_db / 20)

        # Prepare task tuples: (sim_obj, amplitude_high, amplitude_low, snr_low_db, num_trials, idx_power, idx_snr)
        tasks = [
            (self, high_amplitude, amp_low, snr_low_db, num_trials, idx_power, idx_snr)
            for idx_power, amp_low in enumerate(low_amplitudes)
            for idx_snr, snr_low_db in enumerate(snr_lows_db)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _worker_task,
                    sim_obj,
                    amplitude_high,
                    amplitude_low,
                    snr_low_db,
                    num_trials,
                ): (idx_power, idx_snr)
                for sim_obj, amplitude_high, amplitude_low, snr_low_db, num_trials, idx_power, idx_snr in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Simulating",
                mininterval=5.0,
            ):
                idx_power, idx_snr = futures[future]
                p_high, p_low = future.result()
                pdr[0, idx_power, idx_snr] = p_high
                pdr[1, idx_power, idx_snr] = p_low

        return pdr


def _worker_task(
    sim_obj: SimulatorSIC, amp_high: float, amp_low: float, snr_low_db: float, num_trials: int
) -> tuple[float, float]:
    """Worker that runs num_trials for a single (amp_low, snr_low_db)."""
    num_successes_high: int = 0
    num_successes_low: int = 0
    for _ in range(num_trials):
        success_high, success_low = sim_obj.simulate_single_trial(
            amplitude_high=amp_high,
            amplitude_low=amp_low,
            snr_low_db=snr_low_db,
        )
        num_successes_high += int(success_high)
        num_successes_low += int(success_low)
    return num_successes_high / num_trials, num_successes_low / num_trials


if __name__ == "__main__":
    cfg = SimulationConfig(
        sampling_rate=10e6,  # Samples per second
        protocol_high="802154",  # BLE or IEEE 802.15.4
        protocol_low="ble",
        ble_rate=1e6,  # 1 Mb/s or 2 Mb/s
        amplitude_high=10 ** (-6 / 20),  # Amplitude for higher-power signal (fixed)
        amplitude_low=10 ** (-10 / 20),  # Amplitude for lower-power signal (swept)
        snr_low_db=0,  # SNR in (dB) relative to the lower power generated signal
        freq_offset_range=range(-5000, 5000, 50),  # (Hz) For demodulation brute force search
        fine_step=2,  # Step size (Hz) for the fine search
        fine_window=50,  # Half-width (Hz) of the window around best coarse frequency
        payload_len_high=30,  # Bytes in high-power payload
        payload_len_low=200,  # Bytes in low-power payload
        sample_shift_range_low=(0, 1),  # Range to randomly apply fractional delays on IQ data
        sample_shift_range_high=(400, 2200),
        freq_high=None,  # Fixed frequency offset for high-power signal, if None random in freq_offset_range
        freq_low=None,  # Fixed frequency offset for low-power signal, if None random in freq_offset_range
        phase_high=None,  # Fixed phase for high-power signal, if None random
        phase_low=None,  # Fixed phase for low-power signal, if None random
        adc_bits=12,  # ADC resolution in bits
        adc_vmax=1.0,  # Maximum ADC input amplitude
        padding=500,  # Zero-pad the generated signals before adding them to ensure equal length
    )

    simulator = SimulatorSIC(cfg)
    print(simulator.simulate_single_trial(cfg.amplitude_high, cfg.amplitude_low, cfg.snr_low_db, verbose=True))
