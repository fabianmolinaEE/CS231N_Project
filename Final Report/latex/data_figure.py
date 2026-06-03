"""Generate data_figure.pdf: floorplan | power map | thermal label for one instance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import zoom

INSTANCE = "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap"
BASE = "../../data"

fp_raw = np.load(
    f"{BASE}/raw/CircuitNet-N14/routability_features/Vortex-large/Vortex-large/macro_region/{INSTANCE}.npz"
)["data"].astype(np.float32)
pw_raw = np.load(
    f"{BASE}/raw/CircuitNet-N14/IR_drop_features/Vortex-large/Vortex-large/power_all/{INSTANCE}.npz"
)["data"].astype(np.float32)
th = np.load(f"{BASE}/labels/Vortex-large_{INSTANCE.split('Vortex-large_')[1]}/thermal.npy")

TARGET = 256
fp = zoom(fp_raw, (TARGET / fp_raw.shape[0], TARGET / fp_raw.shape[1]), order=1)
pw = zoom(pw_raw, (TARGET / pw_raw.shape[0], TARGET / pw_raw.shape[1]), order=1)

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
fig.subplots_adjust(left=0.02, right=0.96, top=0.88, bottom=0.04, wspace=0.35)

panels = [
    (fp, "Floorplan", "binary_r", False),
    (pw, "Power density", "hot", True),
    (th, "Thermal label (K)", "inferno", True),
]

for ax, (arr, title, cmap, colorbar) in zip(axes, panels):
    vmin, vmax = arr.min(), arr.max()
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    ax.axis("off")
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        cb.outline.set_linewidth(0.5)

plt.savefig("data_figure.pdf", bbox_inches="tight", dpi=150)
plt.savefig("data_figure.png", bbox_inches="tight", dpi=150)
print("Saved data_figure.pdf and data_figure.png")
