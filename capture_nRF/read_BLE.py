import numpy as np
import matplotlib.pyplot as plt

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
samp_rate = 10e6  # 10 MHz

# Compute time axis in microseconds
time = np.arange(len(inst_freq)) / samp_rate * 1e6  # Time in µs

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
plt.figure(figsize=(12, 4))

# Plot instantaneous frequency
plt.plot(time, inst_freq, label="Instantaneous Frequency")
plt.plot(time, binary_inst_freq * 0.18, label="Decoded Bits")

# Add vertical lines and annotate duration
for start, end in zip(start_times, end_times):
    duration = end - start  # Active duration in µs
    plt.axvline(x=start, color="green", linestyle="--", 
                label=f"Start: {start:.2f} µs (Active {duration:.2f} µs)")
    plt.axvline(x=end, color="red", linestyle="--", 
                label=f"End: {end:.2f} µs (Active {duration:.2f} µs)")

plt.title("BLE Packe")
plt.xlabel("Time (µs)")
plt.ylabel("Frequency (rad/sample)")
plt.legend()
plt.tight_layout()
plt.show()
