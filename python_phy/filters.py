import numpy as np
import scipy


# Zero out samples that fall below the amplitude threshold.
def simple_squelch(iq_samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Zero out samples that fall below the amplitude threshold."""
    return np.where(np.abs(iq_samples) < threshold, 0, iq_samples)


# Applies a decimating FIR low-pass filter.
def decimating_fir_filter(
    data: np.ndarray, decimation: int, gain: float, fs: int, cutoff_freq, transition_width, window="hamming"
) -> np.ndarray:
    """Applies a decimating FIR low-pass filter."""
    nyquist = fs / 2
    num_taps = int(4 * nyquist / transition_width)  # Rule of thumb for FIR filter length
    num_taps |= 1  # Ensure odd number of taps

    # Design FIR filter
    taps = scipy.signal.firwin(num_taps, cutoff=cutoff_freq / nyquist, window=window, pass_zero=True)
    taps *= gain  # Apply gain

    # Filter and decimate
    filtered = scipy.signal.lfilter(taps, 1.0, data)
    decimated = filtered[::decimation]  # Downsample
    return decimated


# Apply FIR filtering (convolution) with the given taps
def fir_filter(data: np.ndarray, taps) -> np.ndarray:
    """Apply FIR filtering (convolution) with the given taps."""
    return scipy.signal.convolve(data, taps, mode="same")


# Adds white Gaussian noise to a signal (complex or real)
def add_white_gaussian_noise(signal: np.ndarray, noise_power: float, noise_power_db: bool = True) -> np.ndarray:
    """# Adds white noise to a signal (complex or real)."""
    if noise_power_db:
        noise_power = 10 ** (noise_power / 10)
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(0, np.sqrt(2) / 2, len(signal)) + 1j * np.random.normal(0, np.sqrt(2) / 2, len(signal))
        )
    else:
        noise = np.sqrt(noise_power) * np.random.normal(0, np.sqrt(2) / 2, len(signal))
    return signal + noise


# Single pole IIR filter following GNU Radio implementation.
def single_pole_iir_filter(x, alpha) -> np.ndarray:
    """Single pole IIR filter following GNU Radio implementation."""
    b = [alpha]  # numerator coefficients
    a = [1, -(1 - alpha)]  # denominator coefficients
    return scipy.signal.lfilter(b, a, x)


# Applies a delay to the input data by first applying a fractional delay using an FIR filter,
# and then applying an integer delay via sample shifting with zero-padding.
def fractional_delay_fir_filter(
    data: np.ndarray, delay: float, num_taps: int = 21, same_size: bool = True
) -> np.ndarray:
    """
    Applies a delay to the input data by first applying a fractional delay using an FIR filter,
    and then applying an integer delay via sample shifting with zero-padding.
    """
    # Separate delay into its integer and fractional parts
    integer_delay = int(np.floor(delay))
    fractional_delay = delay - integer_delay

    # Build the FIR filter taps for the fractional delay
    n = np.arange(-num_taps // 2, num_taps // 2)  # ...-3,-2,-1,0,1,2,3...
    fir_kernel = np.sinc(n - fractional_delay)  # Shifted sinc function
    # fir_kernel *= np.hamming(len(n))  # Hamming window (avoid spectral leakage)
    fir_kernel /= np.sum(fir_kernel)  # Normalise filter taps, unity gain
    frac_delayed = scipy.signal.convolve(data, fir_kernel, mode="full")  # Apply filter

    # Compensate for the intrinsic delay caused by convolution
    frac_delayed = np.roll(frac_delayed, -num_taps // 2)
    if same_size:
        frac_delayed = frac_delayed[: len(data)]
    else:
        frac_delayed = frac_delayed[: len(data) + num_taps // 2]

    # Integer delay and pad with zeros
    delayed_output = np.zeros_like(frac_delayed)
    if integer_delay < len(frac_delayed):
        delayed_output[integer_delay:] = frac_delayed[: len(frac_delayed) - integer_delay]

    return delayed_output
