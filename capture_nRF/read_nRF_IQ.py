import numpy as np
import matplotlib.pyplot as plt

# Load complex binary file
data = np.fromfile("data/nrf_IQ.dat", dtype=np.complex64)

start_index = 5000
end_index = 10000
sliced_data = data[start_index:end_index]

# sliced_data.tofile("data/BLE_packet.dat")

plt.plot(np.real(data), label="I")
plt.plot(np.imag(data), label="Q")
plt.legend()
plt.title("Time Domain Signal")
plt.show()

