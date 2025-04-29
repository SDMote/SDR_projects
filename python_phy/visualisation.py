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


def load_and_plot_pkl_data(
    title="PDR vs SNR Analysis", folder="data_pdr", metric: str = "pdr_ratio", figsize: tuple = (10, 6)
):
    """Loads all .pkl files in the specified folder and generates a plot with sequential styling."""
    import os
    import pickle

    # Get all .pkl files in the folder (sorted for consistency)
    pkl_files = sorted([f for f in os.listdir(folder) if f.endswith(".pkl")])

    # Load each .pkl file into a dictionary
    data = {}
    for file in pkl_files:
        file_path = os.path.join(folder, file)
        with open(file_path, "rb") as f:
            data[file] = pickle.load(f)

    # Define plotting parameters
    colours = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray", "cyan", "magenta"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X"]

    # Create an instance of PDRPlotter
    plotter = PDRPlotter()

    # Add each data trace to the plotter with sequentially assigned styles
    for i, (filename, values) in enumerate(data.items()):
        colour = colours[i % len(colours)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        plotter.add_trace(
            values,
            filename,
            colour=colour,
            linestyle=linestyle,
            marker=marker,
            metric=metric,
        )

    # Plot the data
    plotter.plot(title=title, figsize=figsize)


# Plot PDR vs. power difference (high - low) for each signal from a SIC simulation.
def sic_plot_pdr_vs_power(
    high_power_db: float,
    low_powers_db: np.ndarray,  # shape (P,)
    pdr: np.ndarray,  # shape (2, P)
) -> None:
    """
    Plot PDR vs. power difference (high - low) for each signal from a SIC simulation.
    """
    power_diff = high_power_db - low_powers_db
    plt.figure()
    plt.plot(power_diff, pdr[0], marker="o", label="High‐power signal")
    plt.plot(power_diff, pdr[1], marker="s", label="Low‐power signal")
    plt.xlabel("Power Difference (dB)")
    plt.ylabel("Packet Delivery Rate (-)")
    plt.title("PDR vs. Power Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# Plot two heatmaps of PDR over (power difference, snr_lows_db), one per signal.
def sic_plot_pdr_heatmap(
    high_power_db: float,
    low_powers_db: np.ndarray,  # (P,)
    snr_lows_db: np.ndarray,  # (S,)
    pdr: np.ndarray,  # (2, P, S)
    figsize=None,
    max_xticks: int = 6,
    max_yticks: int = 6,
) -> None:
    """
    Plot two heatmaps of PDR over (power difference, snr_lows_db), one per signal.
    """
    nrows, ncols = 1, 2
    sharex, sharey = False, True
    power_diff = high_power_db - low_powers_db
    figsize = figsize if figsize is not None else (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    titles = ["High-power signal", "Low-power signal"]

    def compute_edges(x: np.ndarray) -> np.ndarray:
        """Compute bin edges for imshow."""
        dx = np.diff(x)
        dx = np.hstack([dx[0], dx])  # Assume first spacing same as second
        return np.concatenate([x - dx / 2, [x[-1] + dx[-1] / 2]])

    power_edges = compute_edges(power_diff)
    snr_edges = compute_edges(snr_lows_db)
    vmin, vmax = pdr.min(), pdr.max()  # Common colour scale

    # Reasonable number of ticks for large arrays
    def reduce_ticks(values: np.ndarray, max_ticks: int) -> np.ndarray:
        if len(values) <= max_ticks:
            return values
        indices = np.linspace(0, len(values) - 1, max_ticks, dtype=int)
        return values[indices]

    xticks = reduce_ticks(power_diff, max_xticks)
    yticks = reduce_ticks(snr_lows_db, max_yticks)

    # Plot each heatmap
    ims = []
    for i, ax in enumerate(axes):
        im = ax.imshow(
            pdr[i].T,
            aspect="auto",
            origin="lower",
            extent=[power_edges[0], power_edges[-1], snr_edges[0], snr_edges[-1]],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)

        ax.set_title(titles[i])
        ax.set_xlabel("Power difference (dB)")
        ax.set_xticks(xticks)

        if i == 0:
            ax.set_ylabel("SNR of low-power signal (dB)")
            ax.set_yticks(yticks)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # Fix x-axis direction (low to high values)
        if power_diff[0] > power_diff[-1]:
            ax.invert_xaxis()

    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), location="right", shrink=0.8)
    cbar.set_label("PDR")

    plt.show()


# Dummy test
if __name__ == "__main__":
    # Simulation parameters
    high_power_db = -5.0
    low_powers_db = np.linspace(-20, 0, 20)
    snr_lows_db = np.linspace(-10, 10, 20)
    P, S = len(low_powers_db), len(snr_lows_db)
    # Shape (2, P, S)
    pdr = np.zeros((2, P, S))
    for i in range(2):
        for p in range(P):
            for s in range(S):
                # Just a made-up function
                base = np.clip((snr_lows_db[s] + low_powers_db[p]) / 30 + 0.5, 0, 1)
                pdr[i, p, s] = base * (0.9 if i == 1 else 1.0)

    # 1) Plot a single‐SNR slice vs. power
    snr_idx = 3  # To slice PDR
    pdr_slice = pdr[:, :, snr_idx]  # shape (2, P)
    sic_plot_pdr_vs_power(high_power_db, low_powers_db, pdr_slice)

    # 2) Plot full heatmaps, both layouts
    sic_plot_pdr_heatmap(high_power_db, low_powers_db, snr_lows_db, pdr)

    plt.show()
