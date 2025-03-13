import numpy as np
import scipy


# Modulates a bit sequence by FIR filtering
def pulse_shape_bits_fir(bits: np.ndarray, fir_taps: np.ndarray, sps: int) -> np.ndarray:
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
def modulate_frequency(symbols: np.ndarray, fsk_deviation: float, fs: float) -> np.ndarray:
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


# Generate Gaussian FIR filter taps
def gaussian_fir_taps(sps: int, ntaps: int, bt: float, gain: float = 1.0) -> np.ndarray:
    """Generate Gaussian FIR filter taps"""
    # Scaling factor for time based on BT (bandwidth-bit period product)
    t_scale: float = np.sqrt(np.log(2.0)) / (2 * np.pi * bt)

    # Symmetric time indices around zero
    t = np.linspace(-(ntaps - 1) / 2, (ntaps - 1) / 2, ntaps)

    taps = np.exp(-((t / (sps * t_scale)) ** 2) / 2)  # Gaussian function
    return gain * taps / np.sum(taps)  # Normalise and apply gain


# Generate half-sine pulse FIR filter taps
def half_sine_fir_taps(sps: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, sps + 1))
