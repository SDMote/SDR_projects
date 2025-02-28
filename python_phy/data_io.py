import numpy as np


# Read interleaved float32 values from binary .dat file and convert to complex numbers.
def read_iq_data(filename: str) -> np.ndarray:
    iq = np.fromfile(filename, dtype=np.complex64)
    return iq
