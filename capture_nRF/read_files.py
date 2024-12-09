import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

files = ["BLE_802154_0dB_0dB", # 0
         "BLE_802154_0dB_8dB", # 1
         "BLE_tone_0dB_0dB", # 2
         "BLE_tone_0dB_8dB", # 3
         "802154_BLE_0dB_0dB", # 4
         "802154_BLE_0dB_8dB", # 5
         "802154_tone_0dB_0dB", # 6
         "802154_tone_0dB_8dB", # 7
]

file_num = 7

# Create mask to omit noisy instantaneous frequency
def decode_bits(data, threshold):
    # Apply hysteresis logic
    hysteresis_inst_freq = np.where(inst_freq > threshold, 1, 
                                np.where(inst_freq < -threshold, -1, 0))
    return hysteresis_inst_freq

data = np.fromfile(f"data/{files[file_num]}.dat", dtype=np.complex64)
inst_freq = np.diff(np.unwrap(np.angle(data)))
magnitude = np.abs(data[:-1])

# Sampling rate in Hz
fs = 10e6  # 10 MHz
fLO = 2425e6      # 2425 MHz (local oscillator frequency)

# Compute time axis in microseconds
time = np.arange(len(inst_freq)) / fs * 1e6  # Time in µs

# Create the mask
threshold = np.max(magnitude) * 0.04
mask = magnitude > threshold

# Ignore noise
inst_freq *= mask

# Decode bits
binary_inst_freq = decode_bits(inst_freq, 0.06)

# Plot the results with IQ plots
fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 2, 2, 0.1]})

# Compute time axis in microseconds for IQ plot
time_iq = np.arange(len(data)) / fs * 1e6

# Plot I and Q components
axes[0].plot(time_iq, np.real(data), label="I (In-phase)")
axes[0].plot(time_iq, np.imag(data), label="Q (Quadrature)")
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
if 0 <= file_num <= 3:
    axes[1].set_title("BLE Packet")
else:
    axes[1].set_title("IEEE 802.15.4 Packet")
axes[1].set_xlim(time[0], time[-1])
axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel("Frequency (rad/sample)")
axes[1].legend()
axes[1].grid()

# Compute and plot the spectrogram
f, t, Sxx = spectrogram(data, 
                        fs=fs, 
                        window='hann', 
                        nperseg=256, 
                        noverlap=128, 
                        scaling='density', 
                        mode='complex',
                        return_onesided=False)

# Shift frequencies to include negative values
f = np.fft.fftshift(f - fs / 2)  # Adjust frequency bins for FFT shift
f += fLO + int(fs/2)            # Add fLO to centre the frequency axis
Sxx = np.fft.fftshift(Sxx, axes=0)  # Apply FFT shift to the spectrogram

# Convert to dB scale for visualization
Sxx_dB = 10 * np.log10(np.abs(Sxx))

cmesh = axes[2].pcolormesh(t * 1e6, f / 1e6, Sxx_dB, shading='nearest', cmap='viridis')  # Convert Hz to MHz for display
axes[2].set_title(f"Spectrogram around {int(fLO/1e6)}MHz")
axes[2].set_xlabel("Time (µs)")
axes[2].set_ylabel("Frequency (MHz)")
axes[2].grid()

# Add the colour bar below the spectrogram
cbar = fig.colorbar(cmesh, cax=axes[3], orientation='horizontal')
cbar.set_label('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()
