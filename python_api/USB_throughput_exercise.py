import numpy as np
import adi
import matplotlib.pyplot as plt
import time

'''Goal: measure maximum sample_rate to receive 100% of samples via USB'''

sample_rate = 5e6 # Hz
center_freq = 100e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 100000 # this is the buffer the Pluto uses to buffer samples

times_len = 100
store_times = np.zeros(times_len)
store_samples_len = np.zeros(times_len)
for i in range(times_len):
    start_time = time.time()
    samples = sdr.rx() # receive samples off Pluto
    store_times[i] = time.time() - start_time
    store_samples_len[i] = len(samples)
    
# Print average throughput (samples/s)
print(np.mean(store_samples_len/store_times))
