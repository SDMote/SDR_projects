import numpy as np
import matplotlib.pyplot as plt
import scipy



def read_iq_data(filename: str):
    # Read interleaved float32 values and convert to complex numbers.
    raw = np.fromfile(filename, dtype=np.complex64)
    iq = raw[0::2] + 1j * raw[1::2]
    return iq


def simple_squelch(iq_samples, threshold: float = 0.01):
    # Zero out samples that fall below the amplitude threshold.
    return np.where(np.abs(iq_samples) < threshold, 0, iq_samples)


def low_pass_filter(iq_samples, cutoff, fs: float, numtaps: int = 101):
    # Design a FIR low-pass filter.
    taps = scipy.signal.firwin(numtaps, cutoff / (0.5 * fs))
    filtered = scipy.signal.lfilter(taps, 1.0, iq_samples)
    return filtered


def decimating_fir_filter(data, decimation, gain, fs, cutoff_freq, transition_width, window="hamming"):
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


def plot_iq(ax, data, fs):
    """Plots IQ components on the given axis."""
    time = np.arange(len(data)) / fs * 1e6  # Time in µs
    ax.plot(time, np.real(data), label="I (In-phase)")
    ax.plot(time, np.imag(data), label="Q (Quadrature)")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("IQ Data")
    ax.legend()
    ax.grid()


def plot_time(ax, data_list, fs, labels, title="Time Domain Signal", ylims=None):
    # Plot in time domain
    time = np.arange(len(data_list[0])) / fs * 1e6  # Time in µs
    for data, label in zip(data_list, labels):
        ax.plot(time, np.real(data), label=label)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(time[0], time[-1])
    ax.set_title(title)
    ax.legend()
    ax.grid()
    if ylims:
        ax.set_ylim(ylims)


def plot_spectrogram(ax, data, fs, fLO=0, vmin=-65):
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


def quadrature_demod(iq_samples, gain=1):
    return np.diff(np.unwrap(np.angle(iq_samples))) * gain / 2


def fir_filter(data, taps):
    return np.convolve(data, taps, mode="same")


def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)  # Calculate the noise power based on the SNR (in dB)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


# Start here
filename = "BLE_0dBm"
fs = 10e6  # Hz
fsk_deviation_BLE = 250e3  # Hz
decimation = 1
sps = 10

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


# Plot
def create_subplots(data, fs, fLO=0):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    plot_time(axes[0], [np.real(data[0]), np.imag(data[0])], fs, ["I (In-phase)", "Q (Quadrature)"], "IQ Data")
    cmesh = plot_spectrogram(axes[1], data[0], fs, fLO)
    # fig.colorbar(cmesh, ax=axes[1], label="Power/Frequency (dB/Hz)")
    plot_time(axes[2], [data[1]], fs, ["Instantaneous Frequency"], "Instantaneous Frequency", ylims=(-1.5, 1.5))
    plt.tight_layout()
    plt.show()


create_subplots([iq_samples, freq_samples], fs)
