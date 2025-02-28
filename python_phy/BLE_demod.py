import numpy as np
import matplotlib.pyplot as plt

from data_io import read_iq_data
from filters import simple_squelch, decimating_fir_filter, add_awgn, fir_filter
from visualisation import plot_time, plot_spectrogram
from demodulation import symbol_sync, quadrature_demod, binary_slicer
from packet_utils import correlate_access_code, compute_crc, ble_whitening, binary_to_uint8_array


if __name__ == "__main__":
    filename: str = "BLE_0dBm"
    fs: int | float = 10e6  # Hz
    fsk_deviation_BLE: int | float = 250e3  # Hz
    decimation: int = 1
    sps = 10
    crc_size: int = 3  # 3 bytes CRC for BLE

    # Open file
    iq_samples = read_iq_data(f"../capture_nRF/data/{filename}.dat")

    # iq_samples = add_awgn(iq_samples, snr_db=4)

    # Low pass filter
    iq_samples = decimating_fir_filter(
        iq_samples, decimation=decimation, gain=1, fs=fs, cutoff_freq=1.5e6, transition_width=1000e3, window="hamming"
    )

    # Squelch
    iq_samples = simple_squelch(iq_samples, threshold=10e-2)

    # Quadrature demodulation
    freq_samples = quadrature_demod(iq_samples, gain=(fs / decimation) / (2 * np.pi * fsk_deviation_BLE))

    # Matched filter with instantaneous frequency
    # matched_filter_taps = np.sin(np.linspace(0, np.pi, sps + 1))
    # matched_filter_taps /= np.max(matched_filter_taps)
    # matched = fir_filter(freq_samples, matched_filter_taps)

    # Symbol synchronisation
    synced_samples = symbol_sync(freq_samples, sps=sps)
    synced_samples = binary_slicer(synced_samples)
    preamble_detected = correlate_access_code(
        synced_samples, "01010101_00011110_01101010_00101100_01001000_00000000", threshold=0
    )

    # Packet reading
    for preamble in preamble_detected:
        # Length reading for BLE
        payload_start: int = preamble + 2 * 8
        header = binary_to_uint8_array(synced_samples[preamble:payload_start])  # Whitened
        header, lsfr = ble_whitening(header)  # De-whitened, length_byte includes S0
        payload_length: int = header[-1]  # Payload length in bytes, without CRC

        # Payload reading and de-whitening
        total_bytes: int = payload_length + crc_size
        payload_and_crc = binary_to_uint8_array(synced_samples[payload_start : payload_start + total_bytes * 8])
        payload_and_crc, _ = ble_whitening(payload_and_crc, lsfr)

        # CRC check
        header_and_payload = np.concatenate((header, payload_and_crc[:-crc_size]))
        computed_crc = compute_crc(header_and_payload)
        crc_check = True if (computed_crc == payload_and_crc[-crc_size:]).all() else False

        print(f"{header_and_payload = }")
        print(f"{crc_check = }")

    # Plot
    def create_subplots(data, fs, fLO=0):
        fig, axes = plt.subplots(4, 1, figsize=(10, 6))
        plot_time(
            axes[0], [np.real(data[0]), np.imag(data[0])], fs, ["I (In-phase)", "Q (Quadrature)"], "IQ Data", time=False
        )
        cmesh = plot_spectrogram(axes[1], data[0], fs, fLO)
        # fig.colorbar(cmesh, ax=axes[1], label="Power/Frequency (dB/Hz)")
        plot_time(axes[2], [data[1]], fs, ["Frequency"], "Frequency", ylims=(-1.5, 1.5), time=False)
        plot_time(axes[3], [data[2]], fs, ["Synced"], "Synced", ylims=(-1.5, 1.5), circle=True, time=False)
        plt.tight_layout()
        plt.show()

    create_subplots([iq_samples, freq_samples, synced_samples], fs)
