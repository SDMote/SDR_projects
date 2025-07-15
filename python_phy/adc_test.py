import numpy as np
import matplotlib.pyplot as plt
from receiver import adc_quantise


fs = 500  # Sampling rate (Hz)
f0 = 5  # Signal frequency (Hz)
seconds = 1.0
bits = 3  # ADC resolution
vmax = 1.0
t = np.linspace(0, seconds, int(fs * seconds), endpoint=False)
x = 1 * np.sin(2 * np.pi * f0 * t)
x_adc = adc_quantise(x + 0j, bits=bits, vmax=vmax)

# Plot original waveform
plt.figure()
plt.plot(t, x)
plt.title("Original Sinusoid (0.8V amplitude)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot quantised waveform
plt.figure()
plt.plot(t, x_adc.real)
plt.title(f"Quantized Sinusoid ({bits}-bit ADC, ±{vmax}V)")
plt.xlabel("Time [s]")
plt.ylabel("Quantized Amplitude")
plt.grid(True)

plt.show()
