import numpy as np
import matplotlib.pyplot as plt
import scipy


# Plot in time domain
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
    """Plot in time domain."""
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
        ax.set_xlabel("Time [µs]")
    else:
        ax.set_xlabel("Samples [-]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(time_array[0], time_array[-1])
    ax.set_title(title)
    ax.legend()
    ax.grid()
    if ylims:
        ax.set_ylim(ylims)


# Plots the spectrogram on the given axis.
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


# Hardcoded subplots of time domain and spectrogram
# TODO parametrise
def subplots_iq_spectrogram_bits(data, fs, fLO=0, show=True) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    plot_time(
        axes[0], [np.real(data[0]), np.imag(data[0])], fs, ["I (In-phase)", "Q (Quadrature)"], "IQ Data", time=False
    )
    cmesh = plot_spectrogram(axes[1], data[0], fs, fLO)
    # fig.colorbar(cmesh, ax=axes[1], label="Power/Frequency (dB/Hz)")
    plot_time(axes[2], [data[1]], fs, ["Hard decisions"], "Hard decisions", ylims=(-0.5, 1.5), circle=True, time=False)
    plt.tight_layout()
    if show:
        plt.show()


# Plot each byte of the payload as an unsigned integer
def plot_payload(packet_data: dict) -> None:
    """Plot each byte of the payload as an unsigned integer."""
    plt.figure()
    plt.plot(packet_data["payload"], marker="o", linestyle="-", color="b")
    plt.title("Physical payload")
    plt.xlabel("Byte index")
    plt.ylabel("Bytes as unsigned integers")

    # Create a box with payload details
    crc_text = "OK" if packet_data["crc_check"] else "ERROR"
    info_text = f"Length: {packet_data['length']} B\nCRC: {crc_text} "
    plt.gca().text(
        1.05,
        0.55,
        info_text,
        fontsize=10,
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.grid()
    plt.tight_layout()


# Plots IQ data in vertical subplots.
def subplots_iq(data: list, fs: float, titles=None, labels=None, show=True, figsize=(10, 6)) -> None:
    """Plots IQ data in vertical subplots."""
    num_subplots = len(data)
    _, axes = plt.subplots(num_subplots, 1, figsize=figsize)

    if titles is None:
        titles = [f"IQ Data {i+1}" for i in range(num_subplots)]
    if labels is None:
        labels = [["I (In-phase)", "Q (Quadrature)"]] * num_subplots

    for i, iq_data in enumerate(data):
        plot_time(axes[i], [np.real(iq_data), np.imag(iq_data)], fs, labels[i], titles[i], time=False)

    plt.tight_layout()
    if show:
        plt.show()


# Plots the Bit Error Rate (BER) against frequency offset.
def plot_ber_vs_frequency_offset(freq_range: range, bit_error_rates: np.ndarray, figsize=(8, 5), show=True) -> None:
    """Plots the Bit Error Rate (BER) against frequency offset."""
    plt.figure(figsize=figsize)
    plt.plot(list(freq_range), bit_error_rates, marker="o", linestyle="-", color="b")
    plt.xlabel("Frequency offset [Hz]")
    plt.ylabel("Bit error rate [%]")
    plt.title("BER vs Frequency offset")
    plt.grid(True, linestyle="--", alpha=0.7)
    if show:
        plt.show()


# Plots periodograms of the input signals with shared axes.
def plot_periodograms(
    signals: list[np.ndarray],
    fs: float = 1.0,
    titles: list[str] = None,
    NFFT: int = 1024,
    noverlap: int = 16,
    figsize=(12, 5),
    horizontal: bool = False,
) -> None:
    """Plots periodograms of the input signals with shared axes."""

    n_signals = len(signals)
    if titles is None or len(titles) != n_signals:
        titles = [f"Signal {i+1} Periodogram" for i in range(n_signals)]

    if horizontal:
        _, axes = plt.subplots(1, n_signals, figsize=figsize, sharex=True, sharey=True)
    else:
        _, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=True, sharey=True)

    if n_signals == 1:
        axes = [axes]

    # Plot periodogram for each signal
    for ax, sig, title in zip(axes, signals, titles):
        ax.psd(sig, Fs=fs, NFFT=NFFT, noverlap=noverlap, scale_by_freq=False)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


# Plots the real and imaginary parts of complex signals over time.
def plot_complex_time(
    signals: list[np.ndarray],
    fs: float = 1.0,
    titles: list[str] = None,
    figsize=(12, 5),
    ylim: tuple = None,
    horizontal: bool = False,
) -> None:
    """Plots the real and imaginary parts of complex signals over time."""

    n_signals = len(signals)
    t = 10e6 * np.arange(len(signals[0])) / fs  # Assume all signals have the same length

    # Create default titles if none provided
    if titles is None or len(titles) != n_signals:
        titles = [f"Signal {i+1}" for i in range(n_signals)]

    if horizontal:
        _, axes = plt.subplots(1, n_signals, figsize=figsize, sharex=True, sharey=True)
    else:
        _, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=True, sharey=True)

    if n_signals == 1:
        axes = [axes]

    for ax, sig, title in zip(axes, signals, titles):
        ax.plot(t, np.real(sig), label="Real")
        ax.plot(t, np.imag(sig), label="Imaginary")
        ax.set_title(title)
        ax.set_xlabel("Time (µs)")
        if ylim:
            ax.set_ylim(ylim)

    axes[0].set_ylabel("Amplitude")  # Common y-axis label
    for ax in axes:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


# Compare the Packet Delivery Rate of various demodulation methods in the same plot.
class PDRPlotter:
    """Compare the Packet Delivery Rate of various demodulation methods in the same plot."""

    def __init__(self, x_label: str = "SNR (dB)", y_label: str = "Packet Delivery Rate (-)"):
        """Initialise the plotter with x and y axis labels."""
        self.x_label = x_label
        self.y_label = y_label
        self.results_data = []  # Tuples of (snr_values, metric_values, label, color, linestyle, marker)

    def add_trace(
        self,
        results: dict,
        label: str,
        colour: str = "blue",
        linestyle: str = "-",  # ("-"", "--")
        marker: str = "o",  # ("o", "s", "*")
        metric: str = "pdr_ratio",
    ):
        """Add a result dictionary to be plotted."""
        snr_values = sorted(results.keys())
        metric_values = [results[snr].get(metric, 0) for snr in snr_values]
        self.results_data.append((snr_values, metric_values, label, colour, linestyle, marker))

    def plot(self, title: str = None, legend: bool = True, grid: bool = True, figsize=(10, 6)):
        """Plot all the traces on the same figure."""
        plt.figure(figsize=figsize)

        for snr_values, metric_values, label, color, linestyle, marker in self.results_data:
            plt.plot(snr_values, metric_values, label=label, color=color, linestyle=linestyle, marker=marker)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if grid:
            plt.grid(True)
        plt.show()
