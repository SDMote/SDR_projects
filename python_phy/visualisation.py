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


# Plots the spectrogram on the given axis.
def plot_spectrogram(ax, data: np.ndarray, fs: float | int, fLO: float | int = 0, vmin: float | int = -65):
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
