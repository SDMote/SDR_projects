import os
import numpy as np
from sic_simulator import SimulationConfig


# Read interleaved float32 values from binary .dat file and convert to complex numbers.
def read_iq_data(filename: str) -> np.ndarray:
    iq = np.fromfile(filename, dtype=np.complex64)
    return iq


# Formats protocol: if BLE, specify the transmission rate
def _sic_make_proto_payload_tag(protocol: str, payload_len: int, ble_rate: float) -> str:
    """
    Given a protocol name and payload length, returns:
      - "802154-30B" for non-BLE
      - "BLE1Mbps-200B" for BLE @ 1 Mb/s
    """
    if protocol.upper() == "BLE":
        mbps = int(ble_rate / 1e6)
        protocol = f"BLE{mbps}Mbps"
    return f"{protocol}-{payload_len}B"


# Build a filename to store SIC simulation results.
def sic_make_filename(cfg: SimulationConfig, num_trials: int) -> str:
    """
    Build a filename like:
      802154-30B_BLE1Mbps-200B_20trials.npz
    """
    entries = []
    entries.append(_sic_make_proto_payload_tag(cfg.protocol_high, cfg.payload_len_high, cfg.ble_rate))
    entries.append(_sic_make_proto_payload_tag(cfg.protocol_low, cfg.payload_len_low, cfg.ble_rate))
    entries.append(f"{int(cfg.sampling_rate/1e6)}Msps")
    entries.append(f"{num_trials}trials")

    return "_".join(entries) + ".npz"


# Save SIC simulation results in a .npz file (includes input parameters and results).
def sic_save_simulation(
    cfg: SimulationConfig,
    high_power_db: float,
    low_powers_db: np.ndarray,
    snr_lows_db: np.ndarray,
    num_trials: int,
    pdr: np.ndarray,
    folder: str = "../sic_simulations",
):
    """Save SIC simulation results in a .npz file (includes input parameters and results)."""
    os.makedirs(folder, exist_ok=True)
    fname = sic_make_filename(cfg, num_trials)

    # Dump numpy arrays + cfg.__dict__
    fullpath = os.path.join(folder, fname)
    np.savez_compressed(
        fullpath,
        high_power_db=high_power_db,
        low_powers_db=low_powers_db,
        snr_lows_db=snr_lows_db,
        num_trials=num_trials,
        pdr=pdr,
        cfg=cfg.__dict__,
    )
    print(f"Saved simulation to {fullpath}")
