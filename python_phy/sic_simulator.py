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


@dataclass
class SimulationConfig:
    """Successive Interference Cancellation simulation parameters"""

    sampling_rate: float  # Samples per second
    protocol_high: ReceiverType  # BLE or IEEE 802.15.4
    protocol_low: ReceiverType
    amplitude_high: float  # Amplitude for higher-power signal (fixed at ADC vmax)
    amplitude_low: float  # Amplitude for lower-power signal (swept)
    snr_low_db: float  # SNR in (dB) relative to the lower power generated signal
    freq_offset_range: range  # (Hz) For demodulation brute force search
    fine_step: float | None  # Step size (Hz) for the fine search
    fine_window: float | None  # Half-width (Hz) of the window around best coarse frequency
    payload_len_high: int  # Bytes in high-power payload
    payload_len_low: int  # Bytes in low-power payload
    num_trials: int  # Monte Carlo trials per power difference
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

        # Assign transmitter and receiver classes respectively
        protocol_map = {
            "BLE": (TransmitterBLE, ReceiverBLE),
            "802154": (Transmitter802154, Receiver802154),
        }
        TransmitterClassHigh: Type[Transmitter]
        ReceiverClassHigh: Type[Receiver]
        TransmitterClassLow: Type[Transmitter]
        ReceiverClassLow: Type[Receiver]
        TransmitterClassHigh, ReceiverClassHigh = protocol_map[self.cfg.protocol_high]
        TransmitterClassLow, ReceiverClassLow = protocol_map[self.cfg.protocol_low]

        self.transmitter_high = TransmitterClassHigh(config.sampling_rate)
        self.receiver_high = ReceiverClassHigh(config.sampling_rate)
        self.transmitter_low = TransmitterClassLow(config.sampling_rate)
        self.receiver_low = ReceiverClassLow(config.sampling_rate)

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
        received_packet_high: list[dict] = self.receiver_high.demodulate_to_packet(rx_iq)
        if received_packet_high:
            received_packet_high: dict = received_packet_high[0]
            print(f"{received_packet_high = }")
            success_high = received_packet_high["crc_check"]
        else:
            return False, False  # If the highest power signal is not demodulated, SIC didn't work

        # Synthesise the high-power signal and subtract
        synth_high_iq = self.transmitter_high.modulate_from_payload(received_packet_high["payload"])
        subtracted_iq = subtract_interference_wrapper(
            rx_iq, synth_high_iq, self.cfg.sampling_rate, self.cfg.freq_offset_range, verbose=verbose
        )

        subplots_iq(
            [rx_iq, synth_high_iq, subtracted_iq],
            self.cfg.sampling_rate,
            ["Interfered packet", "Synthesised", "Subtracted"],
            show=True,
        )

        # Demodulate low-power signal
        received_packet_low: list[dict] = self.receiver_low.demodulate_to_packet(subtracted_iq)
        if received_packet_low:
            received_packet_low: dict = received_packet_low[0]
            print(f"{received_packet_low = }")
            success_low = received_packet_low["crc_check"]
        else:
            return success_high, False

        return success_high, success_low


if __name__ == "__main__":
    cfg = SimulationConfig(
        sampling_rate=10e6,  # Samples per second
        protocol_high="802154",  # BLE or IEEE 802.15.4
        protocol_low="BLE",
        amplitude_high=0.9,  # Amplitude for higher-power signal (fixed)
        amplitude_low=0.3,  # Amplitude for lower-power signal (swept)
        snr_low_db=60,  # SNR in (dB) relative to the lower power generated signal
        freq_offset_range=range(-5000, 5000, 50),  # (Hz) For demodulation brute force search
        fine_step=None,  # Step size (Hz) for the fine search
        fine_window=None,  # Half-width (Hz) of the window around best coarse frequency
        payload_len_high=10,  # Bytes in high-power payload
        payload_len_low=100,  # Bytes in low-power payload
        num_trials=1,  # Monte Carlo trials per power difference
        sample_shift_range_low=(0, 1),  # Range to randomly apply fractional delays on IQ data
        sample_shift_range_high=(900, 2200),
        freq_high=None,  # Fixed frequency offset for high-power signal, if None random in freq_offset_range
        freq_low=None,  # Fixed frequency offset for low-power signal, if None random in freq_offset_range
        phase_high=None,  # Fixed phase for high-power signal, if None random
        phase_low=None,  # Fixed phase for low-power signal, if None random
        adc_bits=12,  # ADC resolution in bits
        adc_vmax=1.0,  # Maximum ADC input amplitude
        padding=500,  # Zero-pad the generated signals before adding them to ensure equal length
    )

    simulator = SimulatorSIC(cfg)
    simulator.simulate_single_trial(cfg.amplitude_high, cfg.amplitude_low, cfg.snr_low_db, verbose=True)

