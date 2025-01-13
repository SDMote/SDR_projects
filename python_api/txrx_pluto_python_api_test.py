import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -20 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Create transmit waveform (QPSK, 16 samples per symbol)
num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Receive samples
rx_samples = sdr.rx() # len(rx_samples) = num_samps
print(rx_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# Calculate power spectral density (frequency domain version of signal)
psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))



# Create subplots for cleaner organization
fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

# Time Domain Plot
axes[0].plot(np.real(rx_samples[::1]), label="Real")
axes[0].plot(np.imag(rx_samples[::1]), label="Imaginary")
axes[0].set_title("Time Domain Signal")
axes[0].set_xlabel("Samples")
axes[0].legend()
axes[0].grid()

# Instantaneous Phase Plot
axes[1].plot(np.angle(rx_samples), label="Instantaneous Phase", color="orange")
axes[1].set_title("Instantaneous Phase")
axes[1].set_xlabel("Samples")
axes[1].set_ylabel("Phase [radians]")
axes[1].grid()

# Frequency Domain Plot
axes[2].plot(f / 1e3, psd_dB)
axes[2].set_title("Frequency Domain (PSD)")
axes[2].set_xlabel("Frequency [kHz]")
axes[2].set_ylabel("PSD [dB]")
axes[2].grid()

# I-Q Trajectory Plot
axes[3].plot(np.real(rx_samples), np.imag(rx_samples), label="IQ Trajectory", linewidth=1)
axes[3].set_title("I-Q Plane (Signal Trajectory)")
axes[3].set_xlabel("I (In-phase)")
axes[3].set_ylabel("Q (Quadrature)")
axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")  # Horizontal axis
axes[3].axvline(0, color="black", linewidth=0.5, linestyle="--")  # Vertical axis
axes[3].legend()
axes[3].grid()

# Show all plots
plt.show()
