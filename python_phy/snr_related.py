import numpy as np
import scipy


def qfunc(x: float) -> float:
    return 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))


def theoretical_pdr(SNR_dB: float, PDU_bytes: int = 124) -> float:
    CRC_bytes = 3
    preamble_bytes = 8
    N = (PDU_bytes + CRC_bytes + preamble_bytes) * 8
    SNR = 10 ** (SNR_dB / 10)
    Rb = 1
    B = 1.5 * Rb
    EbN0 = (B / Rb) * SNR
    Pb = qfunc(np.sqrt(2 * EbN0))
    PDR = (1 - Pb) ** N
    return PDR


# Adds white Gaussian noise to a signal (complex or real)
def add_white_gaussian_noise(signal: np.ndarray, noise_power: float, noise_power_db: bool = True) -> np.ndarray:
    """Adds white noise to a signal (complex or real)."""
    if noise_power_db:
        noise_power = 10 ** (noise_power / 10)
    if np.iscomplexobj(signal):
        # Half the power in I and Q components respectively
        noise = np.sqrt(noise_power) * (
            np.random.normal(0, np.sqrt(2) / 2, len(signal)) + 1j * np.random.normal(0, np.sqrt(2) / 2, len(signal))
        )
    else:
        # White Gaussian noise power is equal to its variance
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
    return signal + noise


# DEPRECATED
# Adds white Gaussian noise to the input signal such that the SNR within a
# specified bandwidth centered at zero, capitalising on Parseval's theorem.
# def add_awgn_band_limited(signal: np.ndarray, snr_db: float, fs: float, bw: float) -> np.ndarray:
#     """
#     Adds white Gaussian noise to the input signal such that the SNR within a
#     specified bandwidth centered at zero, capitalising on Parseval's theorem.
#     """
#     N = len(signal)  # signal can be real or complex
#     dft_signal = np.fft.fft(signal)  # Compute FFT
#     freqs = np.fft.fftfreq(N, d=(1 / fs))  # Frequency FFT bins
#     band_inds = np.where(np.abs(freqs) <= bw / 2)[0]  # Frequency FFT bins around zero

#     # Parseval for discrete samples: mean(|x|^2) = sum(|X|^2) / N^2
#     signal_power_band = np.sum(np.abs(dft_signal[band_inds]) ** 2) / (N**2)
#     snr_linear = 10 ** (snr_db / 10)

#     # Here we are scaling the noise power so that the noise power **within the band** yields
#     # the desired SNR in the band. The noise is still generated over the entire sampled bandwidth.
#     # if bw = fs, then the signal power is simply computed over the entire sampled bandwidth, which,
#     # following Parseval's theorem, is equal to np.mean(np.abs(signal)**2) in the time domain.
#     noise_power = (signal_power_band / snr_linear) * (fs / bw)

#     # Generate AWGN with the computed noise power (variance)
#     noise = add_white_gaussian_noise(np.zeros_like(signal), noise_power, noise_power_db=False)

#     return signal + noise


def compute_snr_from_pearson(signal: np.ndarray, noisy_signal: np.ndarray, snr_db: bool = True) -> float:
    pearson = np.abs(np.corrcoef(signal, noisy_signal)[0, 1])
    snr = pearson**2 / (1 - pearson**2)
    return 10 * np.log10(snr) if snr_db else snr


def compute_signal_power(signal: np.ndarray, sample_interval: tuple = (0, -1), power_db: bool = True) -> float:
    signal_present = signal[sample_interval[0] : sample_interval[1]]
    power = np.sum(np.abs(signal_present) ** 2) / len(signal_present)
    return 10 * np.log10(power) if power_db else power


def add_awgn_signal_present(signal: np.ndarray, snr_db: float, sample_interval: tuple = (0, -1)) -> np.ndarray:
    """Add white Gaussian noise, scaling its power relative to the signal's power in the given sample interval."""
    signal_power_db = compute_signal_power(signal, sample_interval)
    noise_power_db = signal_power_db - snr_db

    return add_white_gaussian_noise(signal, noise_power_db)
