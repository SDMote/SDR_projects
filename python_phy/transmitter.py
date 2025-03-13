import numpy as np
import scipy

from modulation import modulate_frequency, pulse_shape_bits_fir
from filters import gaussian_fir_taps
from packet_utils import create_ble_phy_packet, unpack_uint8_to_bits


class TransmitterBLE:
    # Class variables
    transmission_rate: float = 1e6  # BLE 1 Mb/s
    fsk_deviation_ble: float = 250e3  # Hz
    bt: float = 0.5  # Bandwidth-bit period product for Gaussian pulse shaping

    def __init__(self, fs: int | float):
        # Instance variables
        self.fs = fs  # Sampling rate
        # For now, assume sampling rate is an integer multiple of transmission rate.
        # For generalisation, it's necessary to implement a rational resampler to generate the final IQ signal.
        self.sps: int = int(self.fs / self.transmission_rate)  # Samples per symbol

    # Receives a binary array and returns IQ GFSK modulated compplex signal.
    def modulate(self, bits: np.ndarray, zero_padding: int = 0) -> np.ndarray:
        """Receives a binary array and returns IQ GFSK modulated compplex signal."""

        # Generate Gaussian taps and convolve with rectangular window
        gauss_taps = gaussian_fir_taps(sps=self.sps, ntaps=self.sps, bt=self.bt)
        gauss_taps = scipy.signal.convolve(gauss_taps, np.ones(self.sps))

        # Apply Gaussian pulse shaping with BT = 0.5 (BLE PHY specification)
        pulse_shaped_symbols = pulse_shape_bits_fir(bits, fir_taps=gauss_taps, sps=self.sps)

        # Frequency modulation
        iq_signal = modulate_frequency(pulse_shaped_symbols, self.fsk_deviation_ble, self.fs)

        # Append zeros
        iq_signal = np.concatenate(
            (np.zeros(zero_padding, dtype=iq_signal.dtype), iq_signal, np.zeros(zero_padding, dtype=iq_signal.dtype))
        )

        return iq_signal

    # Receive payload (bytes) and base address to create physical BLE packet (bits)
    def process_phy_payload(self, payload: np.ndarray, base_address: int = 0x12345678) -> np.ndarray:
        """Receive payload (bytes) and base address to create physical BLE packet."""

        # Append header, CRC and apply whitening
        byte_packet = create_ble_phy_packet(payload, base_address)

        # Unpack bytes into bits, LSB first as sent on air
        bits_packet = unpack_uint8_to_bits(byte_packet)

        return bits_packet
