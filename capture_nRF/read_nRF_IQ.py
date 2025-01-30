import numpy as np
import matplotlib.pyplot as plt

# Load complex binary file
data = np.fromfile("data/nrf_IQ.dat", dtype=np.complex64)

start_index = 120000
end_index = 135000
sliced_data = data[start_index:end_index]


files = ["BLE_802154_0dB_0dB", # 0
         "BLE_802154_0dB_8dB", # 1
         "BLE_tone_0dB_0dB", # 2
         "BLE_tone_0dB_8dB", # 3
         "802154_BLE_0dB_0dB", # 4
         "802154_BLE_0dB_8dB", # 5
         "802154_tone_0dB_0dB", # 6
         "802154_tone_0dB_8dB", # 7
         "BLE", # 8
         "802154", # 9
         "BLE_whitening", # 10
]

sliced_data.tofile(f"data/{files[10]}.dat")

plt.plot(np.real(data), label="I")
plt.plot(np.imag(data), label="Q")
plt.legend()
plt.title("Time Domain Signal")
plt.show()

