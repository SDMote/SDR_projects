import numpy as np
import scipy


# Modulates a bit sequence by FIR filtering
def modulate_bits_fir(bits: np.ndarray, fir_taps: np.ndarray, sps: int) -> np.ndarray:
    """Modulates a bit sequence by
    1. Upsampling according to samples per symbol (sps).
    2. Filtering with an FIR filter defined by the FIR taps (fir_taps).
    """
    # Upsample with zeros
    upsampled_bits = np.zeros(len(bits) * sps)
    upsampled_bits[::sps] = bits.astype(np.int16) * 2 - 1  # (-1 to 1)

    # Apply FIR filtering and crop at the end (sps - 1) samples
    return scipy.signal.convolve(upsampled_bits, fir_taps, mode="full")[: -sps + 1]


# Modulates in frequency a real array of symbols. Outputs IQ complex signal
def frequency_modulate(symbols: np.ndarray, fsk_deviation: float, fs: float) -> np.ndarray:
    """Modulates in frequency a real array of symbols. Outputs IQ complex signal."""
    # fsk_deviation: a value of 1 in symbols maps to a frequency of fsk_deviation
    # Compute the phase increment per sample based on fsk_deviation
    phase_increments = symbols * (2 * np.pi * fsk_deviation / fs)

    # Prepending a zero ensures that the signal starts at phase 0
    phase_increments = np.insert(phase_increments, 0, 0)

    # Integrate the phase increments
    phase: np.ndarray = np.cumsum(phase_increments)

    # Generate the complex IQ signal as the unit complex exponential of the phase
    iq_signal = np.exp(1j * phase)
    return iq_signal
