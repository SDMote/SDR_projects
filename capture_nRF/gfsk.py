import numpy as np
from math import pi

# ============================================================================
# Frequency Shift Keying Modulation
# Alfonso Cortés - Inria AIO
# Based in the work by Brandon Hippe (PSU)
# 
# ============================================================================

class ModulationGFSK:
    """GFSK digital modulator/demodulator.

    sample_freq: Sampling frequency, at least twice the mark frequency.
    carrier_freq: Carrier (or middle) frequency of the modulation.
    freq_delta: half the distance between mark and space frequencies.
    samples_per_bit: number of samples per bit (symbol).
    adc_depth: number of bits of quantization
    round: rounding rule for quantization
    """

    def __init__(
        self,
        sample_freq,
        carrier_freq,
        freq_delta,
        samples_per_bit,
        quant_bits=4,
        round="round",
    ):
        """Create ModulationGFSK instance.

        sample_freq: Sampling frequency, at least twice the mark frequency.
        carrier_freq: Carrier (or middle) frequency of the modulation.
        freq_delta: half the distance between mark and space frequencies.
        samples_per_bit: number of samples per bit (symbol).
        adc_depth: number of bits of quantization.
        round: rounding rule for quantization (round or truncate or None).
        """

        self.sample_freq = sample_freq
        self.sample_period = 1 / sample_freq
        self.carrier_freq = carrier_freq
        self.freq_delta = freq_delta
        if sample_freq < 2 * (carrier_freq + freq_delta):
            print("The sample frequency is too low")
        self.samples_per_bit = int(samples_per_bit)
        self.bit_freq = sample_freq / samples_per_bit
        self.bit_period = samples_per_bit / sample_freq
        self.quant_bits = quant_bits
        self.max_val = 2 ** (quant_bits - 1) - 1
        self.min_val = -(2 ** (quant_bits - 1))
        self.quant_amp = (2**quant_bits - 1) / 2
        self.quant_off = -0.5
        self.round = round

        bandwidth_time = 2 * freq_delta * 2
        alpha = np.sqrt(np.log(2) / 2) / bandwidth_time
        t = np.linspace(-self.bit_period, self.bit_period, 2 * samples_per_bit + 1)
        self.filter = (np.sqrt(pi) / alpha) * np.exp(-((pi * t / alpha) ** 2))

        self.template_0 = self.modulate("0", amp=2)
        self.template_1 = self.modulate("1", amp=2)

    def filter_bitstream(self, bitstream):
        """Generates sampled digital signal and applies gaussian filter.

        bitstream: string of 1s and 0s.
        return: filtered bitstream.
        """
        M = self.samples_per_bit
        digital = np.repeat([float(int(bit) * 2 - 1) for bit in bitstream], M)
        digital = np.concatenate([digital[:M], digital, digital[-M:]])
        filtered_bitstream = np.convolve(digital, self.filter, mode="valid")
        filtered_bitstream /= np.max(np.abs(filtered_bitstream))
        return filtered_bitstream

    def modulate(self, bitstream: str, initial_phase=0, amp=1):
        """Performs GFSK modulation of the bitstream.

        Each sample is calculated from a linear aproximation of the frequency variation.
        bitstream: string of 1s and 0s.
        intial_phase: phase for the first sample.
        amp: multiply signal by scalar (in addition to quantization amplitude).
        return: GFSK complex signal.
        """
        filtered = self.filter_bitstream(bitstream)
        N = len(filtered)
        frequency = self.freq_delta * (filtered[:-1] + filtered[1:]) / 2
        frequency += self.carrier_freq
        gfsk = np.zeros(N, dtype=complex)
        phase = initial_phase
        gfsk[0] = np.exp(1j * phase)
        for i in range(N - 1):
            phase = phase + 2 * pi * frequency[i] * self.sample_period
            gfsk[i + 1] = np.exp(1j * phase)
        gfsk = self.quant_amp * gfsk + self.quant_off * (1 + 1j)
        gfsk = amp * gfsk
        if self.round:
            if isinstance(self.round, str) and self.round.lower() == "truncate":
                gfsk = np.trunc(gfsk.real) + 1j * np.trunc(gfsk.imag)
            else:
                gfsk = np.round(gfsk)
        gfsk = gfsk - amp * self.quant_off * (1 + 1j)
        return gfsk

    def correlate(self, signal):
        """Calculate cross-correlation between GFSK signal the templates.

        signal: received complex GFSK signal.
        return: cross-correlation with space, mark (starts in the last sample of the first symbol).
        """
        ccor_ii = np.square(np.correlate(signal.real, self.template_0.real))
        ccor_iq = np.square(np.correlate(signal.real, self.template_0.imag))
        ccor_qi = np.square(np.correlate(signal.imag, self.template_0.real))
        ccor_qq = np.square(np.correlate(signal.imag, self.template_0.imag))
        correlation_0 = ccor_ii + ccor_iq + ccor_qi + ccor_qq
        ccor_ii = np.square(np.correlate(signal.real, self.template_1.real))
        ccor_iq = np.square(np.correlate(signal.real, self.template_1.imag))
        ccor_qi = np.square(np.correlate(signal.imag, self.template_1.real))
        ccor_qq = np.square(np.correlate(signal.imag, self.template_1.imag))
        correlation_1 = ccor_ii + ccor_iq + ccor_qi + ccor_qq
        return correlation_0, correlation_1
