import numpy as np
import scipy
from visualisation import compare_bits_with_reference


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
    template_energy = np.sum(np.abs(interference) ** 2)  # For amplitude estimation
    offset = len(interference) - 1  # Because using mode="full"
    correlation = scipy.signal.correlate(affected, interference, mode="full")[offset : offset + len(affected)]
    correlation /= template_energy  # Normalise for amplitude estimation
    return correlation


# Subtract a known interference from an affected packet.
def subtract_interference_wrapper(
    affected: np.ndarray,
    interference: np.ndarray,
    fs: float,
    freq_offsets: list[float],
    amplitude: float = None,
    phase: float = None,
    samples_shift: int = None,
) -> np.ndarray:
    """Subtract a known interference from an affected packet."""
    est_frequency, est_amplitude, est_phase, est_samples_shift = find_interference_parameters(
        affected, interference, freq_offsets, fs
    )
    # Update parameters if not defined
    amplitude, phase, samples_shift = map(
        lambda old, new: new if old is None else old,
        (amplitude, phase, samples_shift),
        (est_amplitude, est_phase, est_samples_shift),
    )

    # Subtract the interference
    ready_to_subtract = multiply_by_complex_exponential(
        interference, fs, freq=est_frequency, phase=phase, amplitude=amplitude
    )
    ready_to_subtract = pad_interference(affected, ready_to_subtract, samples_shift)

    return affected - ready_to_subtract


# Estimate best frequency offset, amplitude, phase and sample shift to subtract from affected packet.
def find_interference_parameters(
    affected: np.ndarray, interference: np.ndarray, freq_offsets: list[float], fs: float
) -> tuple[float, float, float, int]:
    """Estimate best frequency offset, amplitude, phase and sample shift to subtract from affected packet."""
    best_amplitude = -1.0
    best_frequency = None
    best_phase = None
    best_sample_shift = None

    for frequency in freq_offsets:
        rotated_interference = multiply_by_complex_exponential(interference, fs=fs, freq=frequency)
        correlation = correlation_wrapper(affected, rotated_interference)
        max_idx = np.argmax(np.abs(correlation))
        curr_amplitude = np.abs(correlation[max_idx])

        if curr_amplitude > best_amplitude:
            best_amplitude = curr_amplitude
            best_frequency = frequency
            best_phase = np.angle(correlation[max_idx])
            best_sample_shift = max_idx

    return best_frequency, best_amplitude, best_phase, best_sample_shift


# Compute the Bit Error Rate (BER) for a range of frequency offsets.
def compute_ber_vs_frequency(
    freq_range: range,
    affected: np.ndarray,
    interference: np.ndarray,
    fs: float,
    base_address: int,
    reference_packet: dict,
    receiver: object,
) -> np.ndarray:
    """Compute the Bit Error Rate (BER) for a range of frequency offsets."""
    bit_error_rates: np.ndarray = np.empty(len(freq_range))  # Initialise return

    for index, freq in enumerate(freq_range):
        # Subtract interference using the current frequency offset
        subtracted = subtract_interference_wrapper(
            affected=affected, interference=interference, fs=fs, freq_offsets=[freq]
        )
        # Demodulate subtracted IQ data
        bit_samples = receiver.demodulate(subtracted)
        interfered_packet: list[dict] = receiver.process_phy_packet(
            bit_samples, base_address
        )  # Assume BLE receiver (TODO: change hardcoding)
        if interfered_packet:
            interfered_packet: dict = interfered_packet[0]
            bit2bit_difference = compare_bits_with_reference(interfered_packet["payload"], reference_packet["payload"])
            ber = sum(bit2bit_difference) / len(bit2bit_difference) * 100

            bit_error_rates[index] = ber
        else:
            bit_error_rates[index] = np.nan  # Not detected packet (subtraction probably messed up with preamble)

    return bit_error_rates
