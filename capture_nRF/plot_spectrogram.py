import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_spectrograms_1x4(
    files, fs, fLO, save_fig=False, figsize=(16, 5), output_filename="spectrograms.pdf", 
    vlines=None, vspace=0.1, hspace=0.1
):
    """
    Plots spectrograms for specific files in a 1x4 grid with rasterisation for spectrograms.

    Args:
        files (list): List of filenames.
        fs (float): Sampling frequency in Hz.
        fLO (float): Local oscillator frequency in Hz.
        save_fig (bool): Whether to save the figure as a file. Default is False.
        figsize (tuple): Figure size in inches. Default is (16, 5).
        output_filename (str): File name for saving the figure. Default is 'spectrograms.pdf'.
        vlines (list): Optional list of vertical line positions (in µs) for each subplot. Default is None.
        vspace (float): Vertical spacing between subplot rows. Ignored for 1x4 layout.
        hspace (float): Horizontal spacing between subplot columns. Default is 0.1.
    """
    # Create figure and 1x4 grid of subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    indices = [1, 3, 5, 7]  # Files to plot

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=hspace)

    # Iterate over files and positions
    for idx, file_num in enumerate(indices):
        ax = axes[idx]
        
        # Load data
        data = np.fromfile(f"data/{files[file_num]}.dat", dtype=np.complex64)
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(
            data,
            fs=fs,
            window='hann',
            nperseg=256,
            noverlap=128,
            scaling='density',
            mode='complex',
            return_onesided=False,
        )
        
        # FFT shift for correct frequency and time display
        f = np.fft.fftshift(f - fs / 2) + fLO + int(fs / 2)  # Recenter frequency
        Sxx = np.fft.fftshift(Sxx, axes=0)  # Shift spectrogram frequencies
        Sxx_dB = 10 * np.log10(np.abs(Sxx))  # Convert to dB
        
        # Rasterise spectrogram to reduce PDF size
        with plt.rc_context({'path.simplify': True}):
            cmesh = ax.pcolormesh(
                t * 1e6,  # Time in microseconds
                f / 1e6,  # Frequency in MHz
                Sxx_dB,
                shading="nearest",
                cmap="viridis",
                vmin=-65,  # Threshold for better contrast
                rasterized=True,  # Enable rasterisation
            )
        
        # Add optional vertical read lines
        if vlines is not None:
            ax.axvline(vlines[idx], color="red", linestyle="--", label="Read line")
        
        # Title
        ax.set_title(f"({chr(97 + idx)})")  # (a), (b), (c), (d)

        # Label adjustments
        if idx == 0:  # Only the leftmost plot gets the Y-label
            ax.set_ylabel("Frequency (MHz)")
        ax.set_xlabel("Time (µs)")
        ax.grid()

    # Add shared color bar for the spectrograms
    fig.colorbar(
        cmesh, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, label="Power/Frequency (dB/Hz)"
    )

    # Save figure if requested
    if save_fig:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Figure saved as {output_filename}")
    
    # Show the figure
    # plt.show()



files = [
    "BLE_tone_0dB_0dB",    # 2
    "BLE_tone_0dB_8dB",    # 3
    "802154_tone_0dB_0dB", # 6
    "802154_tone_0dB_8dB", # 7
    "BLE_802154_0dB_0dB",  # 0
    "BLE_802154_0dB_8dB",  # 1
    "802154_BLE_0dB_0dB",  # 4
    "802154_BLE_0dB_8dB",  # 5
]

fs = 10e6  # Sampling frequency (10 MHz)
fLO = 2425e6  # Local oscillator frequency (2425 MHz)

# Define vertical lines in µs for each plot
vlines = [50, 100, 200, 300]  # Vertical lines for each subplot

plot_spectrograms_1x4(
    files,
    fs,
    fLO,
    save_fig=True,
    figsize=(12, 2.5),
    output_filename="spectrograms.pdf",
    vlines=None,
    hspace=0.05  # Reduce horizontal spacing between plots
)
