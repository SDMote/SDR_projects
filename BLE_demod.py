import numpy as np
import matplotlib.pyplot as plt
import scipy

from gnuradio import digital, blocks, gr


def read_iq_data(filename: str) -> np.ndarray:
    # Read interleaved float32 values and convert to complex numbers.
    iq = np.fromfile(filename, dtype=np.complex64)
    return iq


def simple_squelch(iq_samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    # Zero out samples that fall below the amplitude threshold.
    return np.where(np.abs(iq_samples) < threshold, 0, iq_samples)


def low_pass_filter(iq_samples: np.ndarray, cutoff, fs: float, numtaps: int = 101) -> np.ndarray:
    # Design a FIR low-pass filter.
    taps = scipy.signal.firwin(numtaps, cutoff / (0.5 * fs))
    filtered = scipy.signal.lfilter(taps, 1.0, iq_samples)
    return filtered


def decimating_fir_filter(
    data: np.ndarray, decimation: int, gain: float, fs: int, cutoff_freq, transition_width, window="hamming"
) -> np.ndarray:
    # Applies a decimating FIR low-pass filter.
    nyquist = fs / 2
    num_taps = int(4 * nyquist / transition_width)  # Rule of thumb for FIR filter length
    num_taps |= 1  # Ensure odd number of taps

    # Design FIR filter
    taps = scipy.signal.firwin(num_taps, cutoff=cutoff_freq / nyquist, window=window, pass_zero=True)
    taps *= gain  # Apply gain

    # Filter and decimate
    filtered = scipy.signal.lfilter(taps, 1.0, data)
    decimated = filtered[::decimation]  # Downsample
    return decimated


def plot_iq(ax, data: np.ndarray, fs: float | int) -> None:
    """Plots IQ components on the given axis."""
    time = np.arange(len(data)) / fs * 1e6  # Time in µs
    ax.plot(time, np.real(data), label="I (In-phase)")
    ax.plot(time, np.imag(data), label="Q (Quadrature)")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("IQ Data")
    ax.legend()
    ax.grid()


def plot_time(
    ax,
    data_list: list,
    fs: float | int,
    labels,
    title: str = "Time Domain Signal",
    ylims=None,
    circle: bool = False,
    time: bool = True,
) -> None:
    # Plot in time domain
    if time:
        time_array = np.arange(len(data_list[0])) / fs * 1e6  # Time in µs
    else:
        time_array = np.arange(len(data_list[0]))  # Samples
    for data, label in zip(data_list, labels):
        if circle:
            ax.plot(time_array, np.real(data), label=label, marker="o")
        else:
            ax.plot(time_array, np.real(data), label=label)

    if time:
        ax.set_xlabel("Time (µs)")
    else:
        ax.set_xlabel("Samples (-)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(time_array[0], time_array[-1])
    ax.set_title(title)
    ax.legend()
    ax.grid()
    if ylims:
        ax.set_ylim(ylims)


def plot_spectrogram(ax, data: np.ndarray, fs: float | int, fLO: float | int = 0, vmin: float | int = -65):
    """Plots the spectrogram on the given axis."""
    f, t, Sxx = scipy.signal.spectrogram(
        data, fs, window="hann", nperseg=256, noverlap=128, mode="complex", return_onesided=False
    )
    f = np.fft.fftshift(f - fs / 2) + fLO + fs // 2

    eps = 1e-12  # Offset to prevent log10(0)
    Sxx_dB = 10 * np.log10(np.abs(np.fft.fftshift(Sxx, axes=0)) + eps)

    cmesh = ax.pcolormesh(t * 1e6, f / 1e6, Sxx_dB, shading="nearest", cmap="viridis", vmin=vmin)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(f"Spectrogram around {int(fLO/1e6)} MHz")
    ax.grid()
    return cmesh


def quadrature_demod(iq_samples: np.ndarray, gain: float | int = 1) -> np.ndarray:
    return np.diff(np.unwrap(np.angle(iq_samples))) * gain


def fir_filter(data: np.ndarray, taps) -> np.ndarray:
    return np.convolve(data, taps, mode="same")


def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)  # Calculate the noise power based on the SNR (in dB)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def symbol_sync(
    input_samples: np.ndarray,
    sps: float,
    TED_gain: float = 1.0,
    loop_BW: float = 0.045,
    damping: float = 1.0,
    max_deviation: float = 1.5,
    out_sps: int = 1,
) -> np.ndarray:
    # Convert NumPy array to GNU Radio format
    src = blocks.vector_source_f(input_samples.tolist(), False, 1, [])

    # TED_MUELLER_AND_MULLER
    # TED_MOD_MUELLER_AND_MULLER
    # TED_ZERO_CROSSING
    # TED_GARDNER
    # TED_EARLY_LATE
    # TED_DANDREA_AND_MENGALI_GEN_MSK
    # TED_MENGALI_AND_DANDREA_GMSK
    # TED_SIGNAL_TIMES_SLOPE_ML
    # TED_SIGNUM_TIMES_SLOPE_ML

    # Instantiate the symbol sync block
    symbol_sync_block = digital.symbol_sync_ff(
        digital.TED_MOD_MUELLER_AND_MULLER,
        sps,
        loop_BW,
        damping,
        TED_gain,
        max_deviation,
        out_sps,
        digital.constellation_bpsk().base(),
        digital.IR_MMSE_8TAP,
        128,
        [],
    )

    sink = blocks.vector_sink_f(1, 1024 * 4)  # Sink to collect the output

    # Connect blocks and run
    tb = gr.top_block()
    tb.connect(src, symbol_sync_block)
    tb.connect(symbol_sync_block, sink)
    tb.run()

    return np.array(sink.data())


def binary_slicer(data) -> np.int8:
    return np.where(data >= 0, 1, 0).astype(np.int8)


def correlate_access_code(data: np.ndarray, access_code: str, threshold: int) -> np.ndarray:
    # access_code: from LSB to MSB (as samples arrive on-air)
    access_code = access_code.replace("_", "")
    code_len = len(access_code)
    access_int = int(access_code, 2)  # Convert the access code (e.g. "10110010") to an integer
    mask = (1 << code_len) - 1  # Create a mask to keep only the last code_len bits.
    data_reg = 0
    positions = []  # Positions where the access code has been found in data

    for i, bit in enumerate(data):
        data_reg = ((data_reg << 1) | (bit & 0x1)) & mask  # Shift in the new bit
        if i + 1 < code_len:  # Start comparing once data_reg is filled
            continue

        # Count the number of mismatched bits between data_reg and the access code
        mismatches = bin((data_reg ^ access_int) & mask).count("1")
        if mismatches <= threshold:
            # Report the position immediately after the access code was found
            positions.append(i + 1)

    return np.array(positions)


def BLE_whitening(data: np.ndarray, lfsr=0x01, polynomial=0x11):
    # Input data must be LSB first (0x7C = 124 must be formatted as 0011 1110)
    # LFSR default value is 0x01 as it is the default value in the nRF DATAWHITEIV register
    # The polynomial default value is 0x11 = 0b001_0001 -> x⁷ + x⁴ + 1 (x⁷ is omitted)
    output = np.empty_like(data)  # Initialise output array

    for idx, byte in enumerate(data):
        whitened_byte = 0
        for bit_pos in range(8):
            # XOR the current data bit with LFSR MSB
            lfsr_msb = (lfsr & 0x40) >> 6  # LFSR is 7-bit, so MSB is at position 6
            data_bit = (byte >> bit_pos) & 1  # Extract current bit
            whitened_bit = data_bit ^ lfsr_msb  # XOR

            # Update the whitened byte
            whitened_byte |= whitened_bit << (bit_pos)

            # Update LFSR
            if lfsr_msb:  # If MSB is 1 (before shifting), apply feedback
                lfsr = (lfsr << 1) ^ polynomial  # XOR with predefined polynomial
            else:
                lfsr <<= 1
            lfsr &= 0x7F  # 0x7F mask to keep ksfr within 7 bits

        output[idx] = whitened_byte

    return output, lfsr


def binary_to_uint8_array(binary: np.ndarray) -> np.ndarray:
    # Pack binary array LSB first ([1,1,1,1,0,0,0,0]) into bytes array ([0x0F])
    # Ensure the binary array length is a multiple of 8
    if len(binary) % 8 != 0:
        raise ValueError(f"The binary list {len(binary)} length must be a multiple of 8.")

    # Convert binary to NumPy array
    binary_array = np.array(binary, dtype=np.uint8)
    binary_array = binary_array.reshape(-1, 8)[:, ::-1]  #  LSB to MSB correction
    uint8_array = np.packbits(binary_array, axis=1).flatten()

    return uint8_array


def compute_crc(data: np.ndarray, crc_init: int = 0x00FFFF, crc_poly: int = 0x00065B, crc_size: int = 3) -> np.ndarray:
    crc_mask = (1 << (crc_size * 8)) - 1  # Mask to n-byte width (0xFFFF for crc_size = 2)

    def swap_nbit(num, n):
        num = num & crc_mask
        reversed_bits = f"{{:0{n * 8}b}}".format(num)[::-1]
        return int(reversed_bits, 2)

    crc_init = swap_nbit(crc_init, crc_size)  # LSB -> MSB
    crc_poly = swap_nbit(crc_poly, crc_size)

    crc = crc_init
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):  # Process each bit
            if crc & 0x01:  # Check the LSB
                crc = (crc >> 1) ^ crc_poly
            else:
                crc >>= 1
            crc &= crc_mask  # Ensure CRC size

    return np.array([(crc >> (8 * i)) & 0xFF for i in range(crc_size)], dtype=np.uint8)


if __name__ == "__main__":
    filename: str = "BLE_0dBm"
    fs: int | float = 10e6  # Hz
    fsk_deviation_BLE: int | float = 250e3  # Hz
    decimation: int = 1
    sps = 10
    crc_size: int = 3  # 3 bytes CRC for BLE

    # Open file
    iq_samples = read_iq_data(f"capture_nRF/data/{filename}.dat")

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
        header, lsfr = BLE_whitening(header)  # De-whitened, length_byte includes S0
        payload_length: int = header[-1]  # Payload length in bytes, without CRC

        # Payload reading and de-whitening
        total_bytes: int = payload_length + crc_size
        payload_and_crc = binary_to_uint8_array(synced_samples[payload_start : payload_start + total_bytes * 8])
        payload_and_crc, _ = BLE_whitening(payload_and_crc, lsfr)

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
