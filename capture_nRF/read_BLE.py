import numpy as np
import matplotlib.pyplot as plt
import pywt

# Create mask to omit noisy instantaneous frequency
def decode_bits(data, threshold):
    # Apply hysteresis logic
    hysteresis_inst_freq = np.where(inst_freq > threshold, 1, 
                                np.where(inst_freq < -threshold, -1, 0))
    return hysteresis_inst_freq

data = np.fromfile("data/BLE_packet.dat", dtype=np.complex64)
inst_freq = np.diff(np.unwrap(np.angle(data)))
magnitude = np.abs(data[:-1])


# Create the mask
threshold = np.max(magnitude) * 0.8
mask = magnitude > threshold

# Ignore noise
inst_freq *= mask

# Decode bits
binary_inst_freq = decode_bits(inst_freq, 0.06)

# Plot the results
plt.figure(figsize=(12, 4))

# Plot instantaneous frequency
yscale = 2
plt.subplot(1, 1, 1)
plt.plot(inst_freq, label="Instantaneous Frequency")
plt.plot(binary_inst_freq * 0.18, label="Decoded bits")
plt.title("BLE packet")
plt.tight_layout()
plt.show()
