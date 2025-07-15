import numpy as np
import scipy
import concurrent.futures
from demodulation import TEDType
from receiver import DemodulationType, Receiver, ReceiverBLE, Receiver802154, ReceiverType
from snr_related import add_awgn_signal_present


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

    # Zero padding at the end
    end_padding = len(iq_samples) - (len(iq_samples_interference) + delay_zero_padding)

    padded_interference = np.concatenate(
        [
            np.zeros(delay_zero_padding, dtype=iq_samples_interference.dtype),
            iq_samples_interference,
            np.zeros(end_padding, dtype=iq_samples_interference.dtype),
        ]
    )

    return padded_interference


# Multiply an input complex exponential.
def multiply_by_complex_exponential(
    input_signal: np.ndarray, fs: float, freq: float, phase: float = 0, amplitude: float = 1, offset: complex = 0
) -> np.ndarray:
    """Multiply an input complex exponential."""
    t = np.arange(input_signal.size) / fs
    complex_cosine = offset + amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))

    return input_signal * complex_cosine


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
    freq_offsets: list[float] | range,
    amplitude: float = None,
    phase: float = None,
    samples_shift: int = None,
    *,
    fine_step: float | None = None,  # Step size (Hz) for the fine search
    fine_window: float | None = None,  # Half-width (Hz) of the window around best coarse frequency
    verbose: bool = False,
) -> np.ndarray:
    """Subtract a known interference from an affected packet."""
    est_frequency, est_amplitude, est_phase, est_samples_shift = find_interference_parameters(
        affected, interference, freq_offsets, fs, fine_step=fine_step, fine_window=fine_window
    )
    if verbose:
        print(f"{est_frequency = } [Hz]")
        print(f"{est_amplitude = :.2f} [-]")
        print(f"{est_phase = :.2f} [rad]")
        print(f"{est_samples_shift = } [samples]")

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
    affected: np.ndarray,
    interference: np.ndarray,
    freq_offsets: list[float] | range,
    fs: float,
    *,
    fine_step: float | None = None,  # Step size (Hz) for the fine search
    fine_window: float | None = None,  # Half-width (Hz) of the window around best coarse frequency
) -> tuple[float, float, float, int]:
    """Estimate best frequency offset, amplitude, phase and sample shift to subtract from affected packet.

    If fine_step and fine_window are not None:
      1. Coarse search over freq_offsets
      2. Fine search around the best coarse frequency within ±fine_window at steps of fine_step
    """

    def _single_search(freq_list):
        best_amp = -np.inf
        best_freq = 0.0
        best_ph = 0.0
        best_idx = 0

        for f in freq_list:
            rotated = multiply_by_complex_exponential(interference, fs=fs, freq=f)
            corr = correlation_wrapper(affected, rotated)
            abs_corr = np.abs(corr)
            idx = np.argmax(abs_corr)
            amp = abs_corr[idx]

            if amp > best_amp:
                best_amp = amp
                best_freq = f
                best_ph = np.angle(corr[idx])
                best_idx = idx

        return best_freq, best_amp, best_ph, best_idx

    if fine_step is None or fine_window is None:
        return _single_search(freq_offsets)

    # Else, do fine search after coarse search
    coarse_freq, _, _, _ = _single_search(freq_offsets)
    low = coarse_freq - fine_window
    high = coarse_freq + fine_window
    fine_freqs = np.arange(low, high, fine_step)
    return _single_search(fine_freqs)


# Compute the Bit Error Rate (BER) for a range of frequency offsets.
def compute_ber_vs_frequency(
    freq_range: range,
    affected: np.ndarray,
    interference: np.ndarray,
    fs: float,
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
        interfered_packet: list[dict] = receiver.process_phy_packet(bit_samples)
        if interfered_packet:
            interfered_packet: dict = interfered_packet[0]
            bit2bit_difference = compare_bits_with_reference(interfered_packet["payload"], reference_packet["payload"])
            if bit2bit_difference is None:  # Payload sizes don't match. Probably due to interference in the preamble
                print(f"Warning: Payload size mismatch at frequency offset {freq} Hz. Skipping BER computation.")
                bit_error_rates[index] = np.nan
                continue
            ber = sum(bit2bit_difference) / len(bit2bit_difference) * 100

            bit_error_rates[index] = ber
        else:
            bit_error_rates[index] = np.nan  # Not detected packet (subtraction probably messed up with preamble)

    return bit_error_rates


# Compare two byte arrays and return a bitwise array of differences
def compare_bits_with_reference(payload: np.ndarray, reference_payload: np.ndarray) -> np.ndarray:
    if payload.shape != reference_payload.shape:
        return None

    error_bits = np.bitwise_xor(reference_payload, payload)  # Compute XOR to get differing bits
    return np.unpackbits(error_bits, bitorder="little")  # Unpack to binary representation


# Process one noise realisation: adds noise, demodulates, and categorises the packet result.
def helper_process_noise_realisation(
    iq_samples: np.ndarray,
    snr: float,
    sample_interval: tuple,
    receiver: Receiver,
    demodulation_type: DemodulationType,
    ted_type: TEDType,
) -> str:
    """Process one noise realisation: adds noise, demodulates, and categorises the packet result."""
    iq_noisy = add_awgn_signal_present(iq_samples, snr_db=snr, sample_interval=sample_interval)

    try:
        received_packets = receiver.demodulate_to_packet(
            iq_noisy, demodulation_type=demodulation_type, ted_type=ted_type
        )
    except ValueError as e:
        if "The binary list" in str(e):
            return "preamble_loss"
        else:
            raise

    if not received_packets:
        return "preamble_loss"
    elif not received_packets[0]["crc_check"]:
        return "crc_failure"
    else:
        return "delivered"


# Computes the PDR and its standard deviation over a range of SNR values using concurrent futures.
def pdr_vs_snr_analysis_parallel(
    iq_samples: np.ndarray,
    snr_range: range,
    sample_interval: tuple,
    fs: float,
    receiver_type: ReceiverType,
    demodulation_type: DemodulationType,
    ted_type: TEDType,
    noise_realisations: int,
) -> dict:
    """Computes the PDR and its standard deviation over a range of SNR values using concurrent futures."""
    receiver_classes: dict[str, type] = {"BLE": ReceiverBLE, "IEEE802154": Receiver802154}
    try:
        # Create a new receiver for each SNR value in the worker below
        receiver_factory = receiver_classes[receiver_type]
    except KeyError:
        raise ValueError(f"Invalid receiver type '{receiver_type}'. Choose from {list(receiver_classes.keys())}")

    results: dict = {}

    for snr in snr_range:
        receiver = receiver_factory(fs)
        delivered_count = 0
        preamble_loss_count = 0
        crc_failure_count = 0

        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    helper_process_noise_realisation,
                    iq_samples,
                    snr,
                    sample_interval,
                    receiver,
                    demodulation_type,
                    ted_type,
                )
                for _ in range(noise_realisations)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result == "delivered":
                    delivered_count += 1
                elif result == "preamble_loss":
                    preamble_loss_count += 1
                elif result == "crc_failure":
                    crc_failure_count += 1

        results[snr] = {
            "pdr_ratio": delivered_count / noise_realisations,
            "preamble_loss_ratio": preamble_loss_count / noise_realisations,
            "crc_failure_ratio": crc_failure_count / noise_realisations,
        }

    return results
