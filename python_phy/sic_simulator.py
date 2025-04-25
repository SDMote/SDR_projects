from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from receiver import Receiver, ReceiverType, ReceiverBLE, Receiver802154, adc_quantise
from transmitter import Transmitter, TransmitterBLE, Transmitter802154
from interference_utils import multiply_by_complex_exponential
from snr_related import add_white_gaussian_noise, add_awgn_signal_present
from filters import fractional_delay_fir_filter


@dataclass
class SimulationConfig:
    """Successive Interference Cancellation simulation parameters"""

    protocol_high: ReceiverType  # BLE or IEEE 802.15.4
    protocol_low: ReceiverType
    amplitude_high: float  # Amplitude for higher-power signal (fixed at ADC vmax)
    amplitude_low: float  # Amplitude for lower-power signal (swept)
    snr_to_low_signal: float  # SNR in (dB) relative to the lower power generated signal
    freq_offset_range: list[float] | range  # (Hz) For demodulation brute force search
    payload_len_high: int  # Bytes in high-power payload
    payload_len_low: int  # Bytes in low-power payload
    sampling_rate: float  # Samples per second
    num_trials: int  # Monte Carlo trials per power difference
    freq_high: float = None  # Fixed frequency offset for high-power signal, if None random in freq_offset_range
    freq_low: float = None  # Fixed frequency offset for low-power signal, if None random in freq_offset_range
    phase_high: float = None  # Fixed phase for high-power signal, if None random
    phase_low: float = None  # Fixed phase for low-power signal, if None random
    adc_bits: int = 12  # ADC resolution in bits
    adc_vmax: float = 1.0  # Maximum ADC input amplitude


class SimulatorSIC:
    def __init__(self, config: SimulationConfig) -> None:
        """
        Wrapper class to simulate PDR against variable power differences.
        """
        self.cfg = config

        # Assign transmitter and receiver classes respectively
        protocol_map = {
            "BLE": (TransmitterBLE, ReceiverBLE),
            "IEEE802154": (Transmitter802154, Receiver802154),
        }
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
        freq_offset = freq_offset if freq_offset is not None else np.random.uniform(*self.cfg.freq_offset_range)
        phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        return multiply_by_complex_exponential(iq, self.cfg.sampling_rate, freq_offset, phase, amplitude)
