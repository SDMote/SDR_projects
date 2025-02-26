import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt


# Define 32-chip sequences for IEEE 802.15.4 symbols
IEEE_802154_CHIP_OCTETS = np.array(
    [
        [0xD9, 0xC3, 0x52, 0x2E],  # 0
        [0xED, 0x9C, 0x35, 0x22],  # 1
        [0x2E, 0xD9, 0xC3, 0x52],  # 2
        [0x22, 0xED, 0x9C, 0x35],  # 3
        [0x52, 0x2E, 0xD9, 0xC3],  # 4
        [0x35, 0x22, 0xED, 0x9C],  # 5
        [0xC3, 0x52, 0x2E, 0xD9],  # 6
        [0x9C, 0x35, 0x22, 0xED],  # 7
        [0x8C, 0x96, 0x07, 0x7B],  # 8
        [0xB8, 0xC9, 0x60, 0x77],  # 9
        [0x7B, 0x8C, 0x96, 0x07],  # 10
        [0x77, 0xB8, 0xC9, 0x60],  # 11
        [0x07, 0x7B, 0x8C, 0x96],  # 12
        [0x60, 0x77, 0xB8, 0xC9],  # 13
        [0x96, 0x07, 0x7B, 0x8C],  # 14
        [0xC9, 0x60, 0x77, 0xB8],  # 15
    ]
)


def bit_to_chips_mapping(frame):
    """Maps an array of bytes to an array of chip octets."""
    chips = np.array([], dtype=np.uint8)
    for byte in frame:
        chips = np.concatenate([chips, IEEE_802154_CHIP_OCTETS[byte >> 4], IEEE_802154_CHIP_OCTETS[byte & 0xF]])
    return chips


def generate_IEEE802154_packet(payload):
    """Add physical header to payload"""
    if len(payload) > 127:
        print("PHY payload must be at most 127 bytes long")
        return
    # Build the packet: preamble sequence, SFD, length, and payload
    preamble = [0x00] * 4
    length = [len(payload)]  # Frame length in bytes
    sfd = [0xA7]
    frame = preamble + sfd + length + payload
    return frame


def unpack_frame_to_positive_negative(frame):
    """Maps 0 to -1."""
    bits = np.unpackbits(np.array(frame, dtype=np.uint8))
    signed_bits = 2 * bits.astype(np.int8) - 1
    return signed_bits


def oqpsk_modulator(frame, sampling_rate, symbol_rate):
    """Generates an IQ-modulated IEEE 802.15.4 O-QPSK signal."""
    samples_per_symbol = int(sampling_rate / symbol_rate)

    # Convert frame to a sequence of +1/-1 symbols
    symbols = unpack_frame_to_positive_negative(frame)

    # Separate I and Q components
    I_symbols = symbols[::2]  # Even bits
    Q_symbols = symbols[1::2]  # Odd bits

    samples_per_symbol = int(sampling_rate / symbol_rate)
    I_upsampled = np.repeat(I_symbols, samples_per_symbol)
    Q_upsampled = np.repeat(Q_symbols, samples_per_symbol)

    # Modulate using HSS (Half-Sine Shaping)
    t = np.linspace(0, np.pi, samples_per_symbol, endpoint=False)
    hss = np.sin(t)
    I_modulated = I_upsampled * np.tile(hss, len(I_upsampled) // samples_per_symbol)
    Q_modulated = Q_upsampled * np.tile(hss, len(Q_upsampled) // samples_per_symbol)

    # Add noise and append extra samples
    # noise = np.random.normal(0, 0.1, size=len(iq_signal)) + 1j * np.random.normal(0, 0.1, size=len(iq_signal))
    # iq_signal_noisy = iq_signal + noise
    # extra_samples = int(len(iq_signal_noisy) * 0.2)
    # iq_signal_noisy = np.concatenate([np.zeros(extra_samples), iq_signal_noisy, np.zeros(extra_samples)])

    return I_upsampled, I_modulated


# Example payload
payload = [0xAA]

# Generate packet
# iq_packet = generate_packet(payload, sampling_rate=10_000_000)
# print("Generated IQ signal:", iq_packet)
# print(generate_IEEE802154_packet(payload))

I_upsampled, I_modulated = oqpsk_modulator(payload, int(10e6), int(1e6))

# Create the 2x1 subplot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Plot I_upsampled in the first subplot
axes[0].stem(I_upsampled)
axes[0].set_title("I_upsampled")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Amplitude")

# Plot I_modulated in the second subplot
axes[1].stem(I_modulated)
axes[1].set_title("I_modulated")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Amplitude")

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
