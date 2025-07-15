import numpy as np
import scipy
from abc import ABC, abstractmethod

from modulation import modulate_frequency, pulse_shape_bits_fir, gaussian_fir_taps, oqpsk_modulate, half_sine_fir_taps
from packet_utils import (
    create_ble_phy_packet,
    unpack_uint8_to_bits,
    create_802154_phy_packet,
    map_nibbles_to_chips,
    split_iq_chips,
)


class Transmitter(ABC):
    # Class variables (overriden in derived classes)
    _max_payload_size: int = None

    @abstractmethod
    def modulate(self, bits_or_chips: np.ndarray, zero_padding: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def process_phy_payload(self, payload: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod  # Wrap previous methods in one call
    def modulate_from_payload(self, *args, **kwargs) -> np.ndarray:
        pass


class TransmitterBLE(Transmitter):
    # Class variables
    _valid_rates = (1e6, 2e6)  # BLE 1Mb/s or 2Mb/s
    _bt: float = 0.5  # Bandwidth-bit period product for Gaussian pulse shaping
    _max_payload_size: int = 255

    def __init__(self, sample_rate: int | float, transmission_rate: float = 1e6):
        # Instance variables
        self.sample_rate = sample_rate  # Sampling rate
        self.transmission_rate: float = transmission_rate  # BLE 1Mb/s or 2Mb/s
        self._fsk_deviation: float = self.transmission_rate * 0.25  # Hz

        # For now, assume sampling rate is an integer multiple of transmission rate.
        # For generalisation, it's necessary to implement a rational resampler to generate the final IQ signal.
        self.sps: int = int(self.sample_rate / self.transmission_rate)  # Samples per symbol

    # Receives a binary array and returns IQ GFSK modulated compplex signal.
    def modulate(self, bits: np.ndarray, zero_padding: int = 0) -> np.ndarray:
        """Receives a binary array and returns IQ GFSK modulated complex signal."""

        # Generate Gaussian taps and convolve with rectangular window
        gauss_taps = gaussian_fir_taps(sps=self.sps, ntaps=self.sps, bt=self._bt)
        gauss_taps = scipy.signal.convolve(gauss_taps, np.ones(self.sps))

        # Apply Gaussian pulse shaping with BT = 0.5 (BLE PHY specification)
        pulse_shaped_symbols = pulse_shape_bits_fir(bits, fir_taps=gauss_taps, sps=self.sps)

        # Frequency modulation
        iq_signal = modulate_frequency(pulse_shaped_symbols, self._fsk_deviation, self.sample_rate)

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

    # Generates IQ data from physical payload
    def modulate_from_payload(
        self, payload: np.ndarray, base_address: int = 0x12345678, zero_padding: int = 0
    ) -> np.ndarray:
        """Generates IQ data from physical payload"""
        bits = self.process_phy_payload(payload, base_address)
        return self.modulate(bits, zero_padding)

    @property
    def transmission_rate(self) -> float:
        return self._transmission_rate

    @transmission_rate.setter
    def transmission_rate(self, rate: float) -> None:
        if rate not in self._valid_rates:
            raise ValueError(f"BLE transmission rate must be one of {self._valid_rates!r}")
        self._transmission_rate = rate
        self._fsk_deviation = self.transmission_rate * 0.25
        self.sps: int = int(self.sample_rate / self.transmission_rate)  # Samples per symbol


class Transmitter802154(Transmitter):
    # Class variables
    _transmission_rate: float = 2e6  # 2 Mchip/s
    _max_payload_size: int = 127

    # Chip mapping for IEEE 802.15.4 O-QPSK DSSS encoding
    chip_mapping: np.ndarray = np.array(
        [
            0xD9C3522E,  # 0
            0xED9C3522,  # 1
            0x2ED9C352,  # 2
            0x22ED9C35,  # 3
            0x522ED9C3,  # 4
            0x3522ED9C,  # 5
            0xC3522ED9,  # 6
            0x9C3522ED,  # 7
            0x8C96077B,  # 8
            0xB8C96077,  # 9
            0x7B8C9607,  # A
            0x77B8C960,  # B
            0x077B8C96,  # C
            0x6077B8C9,  # D
            0x96077B8C,  # E
            0xC96077B8,  # F
        ],
        dtype=np.uint32,
    )

    def __init__(self, sample_rate: int | float):
        # Instance variables
        self.sample_rate = sample_rate
        # For now, assume sampling rate is an integer multiple of transmission rate.
        # For generalisation, it's necessary to implement a rational resampler to generate the final IQ signal.
        self.sps: int = int(2 * self.sample_rate / self._transmission_rate)  # Samples per OQPSK symbol

    # Receives a chip uint32 array and returns IQ O-QPSK modulated complex signal.
    def modulate(self, chips: np.ndarray, zero_padding: int = 0) -> np.ndarray:
        """Receives a chip uint32 array and returns IQ O-QPSK modulated complex signal."""

        I_chips, Q_chips = split_iq_chips(chips)  # Maps a the even and odd chips to I chips and Q chips respectively.
        half_sine_pulse = half_sine_fir_taps(self.sps)  # Generate the half-sine pulse
        iq_signal = oqpsk_modulate(I_chips, Q_chips, half_sine_pulse, self.sps)  # O-QPSK modulation

        # Append zeros
        iq_signal = np.concatenate(
            (np.zeros(zero_padding, dtype=iq_signal.dtype), iq_signal, np.zeros(zero_padding, dtype=iq_signal.dtype))
        )

        return iq_signal

    # Creates physical packet and maps it to an array of uint32 chips. CRC is optional.
    def process_phy_payload(self, payload: np.ndarray, append_crc: bool = True) -> np.ndarray:
        """Creates physical packet and maps it to an array of uint32 chips. CRC is optional."""

        # Append header and optional CRC
        byte_packet = create_802154_phy_packet(payload, append_crc=append_crc)

        # Maps bytes to array of uint32 chips
        chips = map_nibbles_to_chips(byte_packet, self.chip_mapping, return_string=False)

        return chips

    # Generates IQ data from physical payload
    def modulate_from_payload(self, payload: np.ndarray, append_crc: bool = True, zero_padding: int = 0) -> np.ndarray:
        """Generates IQ data from physical payload"""
        chips = self.process_phy_payload(payload, append_crc)
        return self.modulate(chips, zero_padding)
