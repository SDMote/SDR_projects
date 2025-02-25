import numpy as np
import matplotlib.pyplot as plt

# Load complex binary file
data = np.fromfile("data/nrf_IQ.dat", dtype=np.complex64)

start_index = 110000
end_index = 120000
sliced_data = data[start_index:end_index]


files = [
    "BLE_802154_0dB_0dB", # 0
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
    "802154_capture2", # 11
    "802154_short_includeCRC", # 12
    "BLE_0dBm", # 13 ------------ BLE alone -----------------------------
    "BLE_802154_0dBm_8dBm_0MHz", # 14 --- BLE + IEEE 802.15.4 interference
    "802154_8dBm_0MHz", # 15 ------------ IEEE 802.15.4 interference alone
    "BLE_tone_0dBm_8dBm_0MHz", # 16 ----- BLE + tone interference
    "tone_8dBm_0MHz_BLE", # 17 ------- tone interference alone
    "802154_BLE_0dBm", # 18 ----  IEEE 802.15.4 alone -------------------
    "802154_BLE_0dBm_8dBm_0MHz", # 19 --- IEEE 802.15.4 + BLE interference
    "BLE_8dBm_0MHz", # 20 --------------- BLE interference alone
    "802154_tone_0dBm_8dBm_0MHz", # 21 -- IEEE 802.15.4 + tone interference
    "tone_8dBm_0MHz_802154", # 22 ---------- tone interference alone
    "BLE_BLE_2424MHz_2426MHz" # 23 --- double BLE -----------------------
    "802154_802154_2425MHz_2430MHz" # 24 --- double 802154 --------------
]
data_number = 13

# sliced_data.tofile(f"data/{files[data_number]}.dat")

plt.plot(np.real(data), label="I")
plt.plot(np.imag(data), label="Q")
plt.legend()
plt.title("Time Domain Signal")
plt.show()

