import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Create mask to omit noisy instantaneous frequency
def decode_bits(data, threshold):
    # Apply hysteresis logic
    hysteresis_inst_freq = np.where(inst_freq > threshold, 1, 
                                np.where(inst_freq < -threshold, -1, 0))
    return hysteresis_inst_freq

data = np.fromfile("data/BLE_packet.dat", dtype=np.complex64)
inst_freq = np.diff(np.unwrap(np.angle(data)))
magnitude = np.abs(data[:-1])

# Sampling rate in Hz
fs = 10e6  # 10 MHz
fLO = 2425e6      # 2425 MHz (local oscillator frequency)

# Compute time axis in microseconds
time = np.arange(len(inst_freq)) / fs * 1e6  # Time in µs

# Create the mask
threshold = np.max(magnitude) * 0.8
mask = magnitude > threshold

# Ignore noise
inst_freq *= mask

# Decode bits
binary_inst_freq = decode_bits(inst_freq, 0.06)

# Find transitions (start and end of active mask regions)
start_times = time[np.where((mask[1:] == 1) & (mask[:-1] == 0))[0] + 1]  # Rising edges
end_times = time[np.where((mask[1:] == 0) & (mask[:-1] == 1))[0] + 1]    # Falling edges

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 2, 0.1]})

# Plot instantaneous frequency
axes[0].plot(time, inst_freq, label="Instantaneous Frequency")
axes[0].plot(time, binary_inst_freq * 0.18, label="Decoded Bits")

# Set limits to align edges with data
axes[0].set_xlim(time[0], time[-1])

# Add vertical lines and annotate duration
for start, end in zip(start_times, end_times):
    duration = end - start  # Active duration in µs
    axes[0].axvline(x=start, color="green", linestyle="--", 
                    label=f"Start: {start:.2f} µs (Active {duration:.2f} µs)")
    axes[0].axvline(x=end, color="red", linestyle="--", 
                    label=f"End: {end:.2f} µs (Active {duration:.2f} µs)")

axes[0].set_title("BLE Packet")
axes[0].set_xlabel("Time (µs)")
axes[0].set_ylabel("Frequency (rad/sample)")
axes[0].legend()
axes[0].grid()

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
f += fLO + int(fs/2)                                # Add fLO to centre the frequency axis
Sxx = np.fft.fftshift(Sxx, axes=0)     # Apply FFT shift to the spectrogram

# Convert to dB scale for visualization
Sxx_dB = 10 * np.log10(np.abs(Sxx))

cmesh = axes[1].pcolormesh(t * 1e6, f / 1e6, Sxx_dB, shading='nearest', cmap='viridis')  # Convert Hz to MHz for display
axes[1].set_title("Spectrogram of Complex Signal (Negative and Positive Frequencies)")
axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel("Frequency (MHz)")
axes[1].grid()

# Add the colour bar below the spectrogram
cbar = fig.colorbar(cmesh, cax=axes[2], orientation='horizontal')
cbar.set_label('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()
