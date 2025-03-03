import numpy as np
import matplotlib.pyplot as plt

from demodulation import symbol_sync, quadrature_demod, binary_slicer
from packet_utils import (
    correlate_access_code,
    compute_crc,
    ble_whitening,
    binary_to_uint8_array,
    generate_access_code_ble,
)


class ReceiverBLE:
    def __init__(self, fs: float, sps: int):
        self.fs = fs  # Sampling rate
        self.sps = sps
        self.crc_size: int = 3  # 3 bytes CRC for BLE
        self.fsk_deviation_ble: int | float = 250e3  # Hz

    # Receives an array of complex data and returns hard decision array
    def demodulate(self, iq_samples: np.ndarray) -> np.ndarray:
        """Receives an array of complex data and returns hard decision array."""
        # Quadrature demodulation
        freq_samples = quadrature_demod(iq_samples, gain=(self.fs) / (2 * np.pi * self.fsk_deviation_ble))

        # Matched filter
        # TODO, something like the following
        # matched_filter_taps = np.sin(np.linspace(0, np.pi, sps + 1))
        # matched_filter_taps /= np.max(matched_filter_taps)
        # matched = fir_filter(freq_samples, matched_filter_taps)

        ## Symbol synchronisation
        bit_samples = symbol_sync(freq_samples, sps=self.sps)
        bit_samples = binary_slicer(bit_samples)
        return bit_samples

    # Receive hard decisions (bit samples) and return dictionary with detected packets
    def process_phy_packet(self, bit_samples: np.ndarray, base_address: int) -> list[dict]:
        """Receive hard decisions (bit samples) and return dictionary with detected packets."""
        # Decode detected packets found in bit_samples array
        preamble_detected = correlate_access_code(bit_samples, generate_access_code_ble(base_address), threshold=1)
        detected_packets: list[dict] = []

        # Read packets starting from the end of the preamble
        for preamble in preamble_detected:
            # Length reading for BLE
            payload_start: int = preamble + 2 * 8  # S0 + length byte
            header = binary_to_uint8_array(bit_samples[preamble:payload_start])  # Whitened
            header, lsfr = ble_whitening(header)  # De-whitened, length_byte includes S0
            payload_length: int = header[-1]  # Payload length in bytes, without CRC

            # Payload reading and de-whitening
            total_bytes: int = payload_length + self.crc_size
            payload_and_crc = binary_to_uint8_array(bit_samples[payload_start : payload_start + total_bytes * 8])
            payload_and_crc, _ = ble_whitening(payload_and_crc, lsfr)

            # CRC check
            header_and_payload = np.concatenate((header, payload_and_crc[: -self.crc_size]))
            computed_crc = compute_crc(header_and_payload)
            crc_check = True if (computed_crc == payload_and_crc[-self.crc_size :]).all() else False
            # print(f"{header_and_payload = }")
            # print(f"{crc_check = }")

            phy_payload = header_and_payload[2:]  # Remove CRC bytes

            # Append dictionary to return list
            detected_packets.append({"phy_payload": phy_payload, "crc_check": crc_check})

        return detected_packets
