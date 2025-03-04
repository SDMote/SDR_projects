import numpy as np
import matplotlib.pyplot as plt

from demodulation import symbol_sync, quadrature_demod, binary_slicer
from filters import single_pole_iir_filter
from packet_utils import (
    correlate_access_code,
    compute_crc,
    ble_whitening,
    pack_bits_to_uint8,
    generate_access_code_ble,
    map_nibbles_to_chips,
    pack_chips_to_bytes,
)


class ReceiverBLE:
    def __init__(self, fs: int | float, sps: int | float):
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
    def process_phy_packet(self, bit_samples: np.ndarray, base_address: int, preamble_threshold: int = 2) -> list[dict]:
        """Receive hard decisions (bit samples) and return dictionary with detected packets."""
        # Decode detected packets found in bit_samples array
        preamble_positions: np.ndarray = correlate_access_code(
            bit_samples, generate_access_code_ble(base_address), threshold=preamble_threshold
        )
        detected_packets: list[dict] = []

        # Read packets starting from the end of the preamble
        for preamble in preamble_positions:
            # Length reading for BLE
            payload_start: int = preamble + 2 * 8  # S0 + length byte
            header = pack_bits_to_uint8(bit_samples[preamble:payload_start])  # Whitened
            header, lsfr = ble_whitening(header)  # De-whitened, length_byte includes S0
            payload_length: int = header[-1]  # Payload length in bytes, without CRC

            # Payload reading and de-whitening
            total_bytes: int = payload_length + self.crc_size
            payload_and_crc = pack_bits_to_uint8(bit_samples[payload_start : payload_start + total_bytes * 8])
            payload_and_crc, _ = ble_whitening(payload_and_crc, lsfr)

            # CRC check
            header_and_payload = np.concatenate((header, payload_and_crc[: -self.crc_size]))
            computed_crc = compute_crc(header_and_payload, crc_init=0x00FFFF, crc_poly=0x00065B, crc_size=self.crc_size)
            crc_check = True if (computed_crc == payload_and_crc[-self.crc_size :]).all() else False
            # print(f"{header_and_payload = }")
            # print(f"{crc_check = }")

            payload = header_and_payload[2:]  # Remove CRC bytes

            # Append dictionary to return list
            detected_packets.append(
                {"payload": payload, "length": len(payload), "crc_check": crc_check, "position_in_array": payload_start}
            )

        return detected_packets


class Receiver802154:
    CHIP_MAPPING: np.ndarray = np.array(
        [
            0xE077AE6C,  # 0
            0xCE077AE6,  # 1
            0x6CE077AE,  # 2
            0xE6CE077A,  # 3
            0xAE6CE077,  # 4
            0x7AE6CE07,  # 5
            0x77AE6CE0,  # 6
            0x877AE6CE,  # 7
            0x1F885193,  # 8
            0x31F88519,  # 9
            0x931F8851,  # A
            0x1931F885,  # B
            0x51931F88,  # C
            0x851931F8,  # D
            0x8851931F,  # E
            0x78851931,  # F
        ],
        dtype=np.uint32,
    )

    def __init__(self, fs: int | float, sps: int | float):
        self.fs = fs  # Sampling rate
        self.sps = sps
        self.crc_size: int = 2  # 2 bytes CRC for IEEE 802.15.4
        self.fsk_deviation_802154: int | float = 500e3  # Hz
        self.max_packet_len: int = 127

    # Receives an array of complex data and returns hard decision array
    def demodulate(self, iq_samples: np.ndarray) -> np.ndarray:
        """Receives an array of complex data and returns hard decision array."""
        # Quadrature demodulation
        freq_samples = quadrature_demod(iq_samples, gain=(self.fs) / (2 * np.pi * self.fsk_deviation_802154))
        freq_samples -= single_pole_iir_filter(freq_samples, alpha=160e-6)

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
    def process_phy_packet(
        self, chip_samples: np.ndarray, preamble_threshold: int = 12, CRC_included: bool = True
    ) -> list[dict]:
        """Receive hard decisions (bit samples) and return dictionary with detected packets."""
        # Decode detected packets found in bit_samples array
        access_code: str = map_nibbles_to_chips([0x00, 0x00, 0x00, 0x00, 0xA7], self.CHIP_MAPPING)
        preamble_positions: np.ndarray = correlate_access_code(
            chip_samples, access_code, threshold=preamble_threshold, reduce_mask=True
        )
        detected_packets: list[dict] = []

        # Read packets starting from the end of the preamble
        for preamble in preamble_positions:
            # Length reading for IEEE 802.15.4
            payload_start: int = preamble + 2 * 32  # 2 nibbles, 1 byte
            payload_length = pack_chips_to_bytes(
                chip_samples[preamble:payload_start], num_bytes=1, chip_mapping=self.CHIP_MAPPING, threshold=10
            )  # Payload length in bytes
            payload_length = payload_length[0]
            assert payload_length <= self.max_packet_len, "Maximum payload length is 127 bytes"

            # Payload reading
            payload = pack_chips_to_bytes(
                chip_samples[payload_start : payload_start + payload_length * 64],
                num_bytes=payload_length,
                chip_mapping=self.CHIP_MAPPING,
                threshold=32,  # Once the preamble and length are detected, just return closest guess
            )

            crc_check = None
            # CRC check
            if CRC_included:
                computed_crc = compute_crc(
                    payload[: -self.crc_size], crc_init=0x0000, crc_poly=0x011021, crc_size=self.crc_size
                )
                crc_check = True if (computed_crc == payload[-self.crc_size :]).all() else False
                payload = payload[: -self.crc_size]  # Remove CRC bytes

            # Append dictionary to return list
            detected_packets.append(
                {"payload": payload, "length": len(payload), "crc_check": crc_check, "position_in_array": payload_start}
            )

        return detected_packets
