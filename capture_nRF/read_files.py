import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# List of available files
files = [
    "BLE_802154_0dB_0dB",  # 0
    "BLE_802154_0dB_8dB",  # 1
    "BLE_tone_0dB_0dB",  # 2
    "BLE_tone_0dB_8dB",  # 3
    "802154_BLE_0dB_0dB",  # 4
    "802154_BLE_0dB_8dB",  # 5
    "802154_tone_0dB_0dB",  # 6
    "802154_tone_0dB_8dB",  # 7
    "BLE",  # 8
    "802154",  # 9
    "BLE_whitening",  # 10
    "802154_capture2",  # 11
    "802154_short_includeCRC",  # 12
    "BLE_0dBm",  # 13 ------------ BLE alone -----------------------------
    "BLE_802154_0dBm_8dBm_0MHz",  # 14 --- BLE + IEEE 802.15.4 interference
    "802154_8dBm_0MHz",  # 15 ------------ IEEE 802.15.4 interference alone
    "BLE_tone_0dBm_8dBm_0MHz",  # 16 ----- BLE + tone interference
    "tone_8dBm_0MHz_BLE",  # 17 ------- tone interference alone
    "802154_0dBm",  # 18 ----  IEEE 802.15.4 alone -------------------
    "802154_BLE_0dBm_8dBm_0MHz",  # 19 --- IEEE 802.15.4 + BLE interference
    "BLE_8dBm_0MHz",  # 20 --------------- BLE interference alone
    "802154_tone_0dBm_8dBm_0MHz",  # 21 -- IEEE 802.15.4 + tone interference
    "tone_8dBm_0MHz_802154",  # 22 ---------- tone interference alone
    "BLE_BLE_2424MHz_2426MHz",  # 23 --- double BLE -----------------------
    "802154_802154_2425MHz_2430MHz",  # 24 --- double 802154 --------------
    "nrf_IQ",  # -1
]


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process and plot FSK signal from file.")
parser.add_argument("-f", "--file_num", type=int, required=True, help="File number to process (0-7)")
args = parser.parse_args()

# Validate file_num
file_num = args.file_num
if file_num < 0 or file_num >= len(files):
    raise ValueError(f"Invalid file number: {file_num}. Must be between 0 and {len(files)-1}.")


# Create mask to omit noisy instantaneous frequency
def decode_bits(data, threshold):
    # Apply hysteresis logic
    hysteresis_inst_freq = np.where(inst_freq > threshold, 1, np.where(inst_freq < -threshold, -1, 0))
    return hysteresis_inst_freq


# Load data
data = np.fromfile(f"data/new/{files[file_num]}.dat", dtype=np.complex64)
print(len(data))
phase_shift = -2 + np.pi / 2 + np.pi
phased_data = data * np.exp(1j * phase_shift)

inst_freq = np.diff(np.unwrap(np.angle(data)))
magnitude = np.abs(data[:-1])

# Sampling rate in Hz
fs = 10e6  # 10 MHz
fLO = 2425e6  # 2425 MHz (local oscillator frequency)

# Compute time axis in microseconds
time = np.arange(len(inst_freq)) / fs * 1e6  # Time in µs

# Create the mask
threshold = np.max(magnitude) * 0.4
mask = magnitude > threshold

# Ignore noise
inst_freq *= mask

# Decode bits
binary_inst_freq = decode_bits(inst_freq, 0.06)

# Plot the results with IQ plots
fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.5, 2, 2, 0.1]})

# Compute time axis in microseconds for IQ plot
time_iq = np.arange(len(data)) / fs * 1e6

# Plot I and Q components
# axes[0].plot(time_iq, np.real(data), label="I (In-phase)")
# axes[0].plot(time_iq, np.imag(data), label="Q (Quadrature)")
axes[0].plot(time_iq, np.real(phased_data), label="I (In-phase)")
axes[0].plot(time_iq, np.imag(phased_data), label="Q (Quadrature)")
# axes[0].plot(time_iq, np.angle(data), label="Phase")
axes[0].set_title("Time Domain Signal (I/Q Components)")
axes[0].set_xlim(time[0], time[-1])
axes[0].set_xlabel("Time (µs)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

# Plot instantaneous frequency and decoded bits
axes[1].plot(time, inst_freq, label="Instantaneous Frequency")
axes[1].plot(time, binary_inst_freq * 0.18, label="Decoded Bits")
axes[1].set_ylabel("Frequency (rad/sample)")
if 0 <= file_num <= 3 or file_num == 8:
    axes[1].set_title("BLE Packet")
else:
    axes[1].set_title("IEEE 802.15.4 Packet")
axes[1].set_xlim(time[0], time[-1])
axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel("Frequency (rad/sample)")
axes[1].legend()
axes[1].grid()

# Compute and plot the spectrogram
f, t, Sxx = spectrogram(
    data, fs=fs, window="hann", nperseg=256, noverlap=128, scaling="density", mode="complex", return_onesided=False
)

# Shift frequencies to include negative values
f = np.fft.fftshift(f - fs / 2)  # Adjust frequency bins for FFT shift
f += fLO + int(fs / 2)  # Add fLO to centre the frequency axis
Sxx = np.fft.fftshift(Sxx, axes=0)  # Apply FFT shift to the spectrogram

# Convert to dB scale for visualization
Sxx_dB = 10 * np.log10(np.abs(Sxx))

cmesh = axes[2].pcolormesh(
    t * 1e6,
    f / 1e6,  # Convert Hz to MHz for display
    Sxx_dB,
    shading="nearest",
    cmap="viridis",
    vmin=-65,  # Apply thresholds for better contrast
)

axes[2].set_title(f"Spectrogram around {int(fLO/1e6)}MHz")
axes[2].set_xlabel("Time (µs)")
axes[2].set_ylabel("Frequency (MHz)")
axes[2].grid()

# Add the colour bar below the spectrogram
cbar = fig.colorbar(cmesh, cax=axes[3], orientation="horizontal")
cbar.set_label("Power/Frequency (dB/Hz)")

plt.tight_layout()
plt.show()


# Plot IQ trajectory in the I-Q plane

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.real(data), np.imag(data), label="IQ Trajectory", linewidth=1)
ax.set_title("I-Q Plane (Signal Trajectory)")
ax.set_xlabel("I (In-phase)")
ax.set_ylabel("Q (Quadrature)")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Horizontal axis
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Vertical axis
ax.legend()
ax.grid()
plt.show()
