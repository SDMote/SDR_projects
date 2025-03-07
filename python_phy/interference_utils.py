import numpy as np
import scipy


# Applies a delay to the input data by first applying a fractional delay using an FIR filter,
# and then applying an integer delay via sample shifting with zero-padding.
def fractional_delay_fir_filter(data: np.ndarray, delay: float, num_taps: int = 21) -> np.ndarray:
    """
    Applies a delay to the input data by first applying a fractional delay using an FIR filter,
    and then applying an integer delay via sample shifting with zero-padding.
    """
    # Separate delay into its integer and fractional parts
    integer_delay = int(np.floor(delay))
    fractional_delay = delay - integer_delay

    # Build the FIR filter taps for the fractional delay
    n = np.arange(-num_taps // 2, num_taps // 2)  # ...-3,-2,-1,0,1,2,3...
    h = np.sinc(n - fractional_delay)  # Shifted sinc function
    h *= np.hamming(len(n))  # Hamming window (avoid spectral leakage)
    h /= np.sum(h)  # Normalise filter taps, unity gain
    frac_delayed = scipy.signal.convolve(data, h, mode="full")  # Apply filter

    # Compensate for the intrinsic delay caused by convolution
    frac_delayed = np.roll(frac_delayed, -num_taps // 2)
    frac_delayed = frac_delayed[: len(data)]

    # Integer delay and pad with zeros
    delayed_output = np.zeros_like(frac_delayed)
    if integer_delay < len(frac_delayed):
        delayed_output[integer_delay:] = frac_delayed[: len(frac_delayed) - integer_delay]

    return delayed_output


# Pads iq_samples_interference with zeros at the beginning (delay_zero_padding)
# and at the end to match the length of iq_samples.
def pad_interference(
    iq_samples: np.ndarray, iq_samples_interference: np.ndarray, delay_zero_padding: int
) -> np.ndarray:
    """
    Pads iq_samples_interference with zeros at the beginning (delay_zero_padding)
    and at the end to match the length of iq_samples.
    """
    assert 0 <= delay_zero_padding < len(iq_samples), "delay_zero_padding out of range"

    # If the interference plus delay is too long, crop the interference
    max_interference_length = len(iq_samples) - delay_zero_padding
    if len(iq_samples_interference) > max_interference_length:
        iq_samples_interference = iq_samples_interference[:max_interference_length]

    # Compute the necessary zero padding at the end
    end_padding = len(iq_samples) - (len(iq_samples_interference) + delay_zero_padding)

    padded_interference = np.concatenate(
        [
            np.zeros(delay_zero_padding, dtype=iq_samples_interference.dtype),  # Front padding
            iq_samples_interference,  # Interference signal
            np.zeros(end_padding, dtype=iq_samples_interference.dtype),  # End padding
        ]
    )

    return padded_interference


# Multiply an input complex exponential.
def multiply_by_complex_exponential(
    input_signal: np.ndarray, fs: float, freq: float, phase: float = 0, amplitude: float = 1, offset: complex = 0
) -> np.ndarray:
    """Multiply an input complex exponential."""
    t = np.arange(input_signal.size) / fs  # Time vector
    complex_cosine = offset + amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))

    return input_signal * complex_cosine  # Return multiplication


# Correlation wrapper to estimate where an interference is on an affected packet.
def correlation_wrapper(affected: np.ndarray, interference: np.ndarray) -> np.ndarray:
    """Correlation wrapper to estimate where an interference is on an affected packet."""
    offset = len(interference) - 1  # Because using mode="full"
    correlation = scipy.signal.correlate(affected, interference, mode="full")[offset : offset + len(affected)]
    return correlation
