import numpy as np
import os

from sic_simulator import SimulationConfig
from visualisation import sic_plot_pdr_heatmap, sic_plot_pdr_vs_power


filename = "BLE1Mbps-200B_802154-30B_10Msps_1000trials.npz"
filename = "802154-30B_BLE1Mbps-200B_10Msps_1000trials.npz"
filename = "802154-30B_BLE1Mbps-200B_4Msps_1000trials.npz"
filename = "802154-30B_BLE1Mbps-200B_30Msps_1000trials.npz"

data = np.load(f"./sic_simulations/{filename}", allow_pickle=True)

# Extract each parameter
high_power_db = float(data["high_power_db"])
low_powers_db = data["low_powers_db"]
snr_lows_db = data["snr_lows_db"]
num_trials = int(data["num_trials"])
pdr = data["pdr"]
cfg_dict = data["cfg"].item()  # Reconstruct the config data class from dictionary
cfg = SimulationConfig(**cfg_dict)

print("PDR shape:", pdr.shape)


print(f"{low_powers_db = }")
print(f"{snr_lows_db = }")
print(pdr.shape)
sic_plot_pdr_vs_power(high_power_db, low_powers_db, pdr[:, :, 7], figsize=None)
sic_plot_pdr_heatmap(
    high_power_db, low_powers_db, snr_lows_db, pdr, figsize=(6, 3), saveas=os.path.splitext(filename)[0] + ".pdf"
)
