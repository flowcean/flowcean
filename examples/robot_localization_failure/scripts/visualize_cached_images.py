from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

# Path to your training cache
CACHE_DIR = "learning_cache"

# Detect number of samples
cache_path = Path(CACHE_DIR)
pt_files = sorted(cache_path.glob("*.pt"), key=lambda p: int(p.stem))

if len(pt_files) == 0:
    raise RuntimeError("No .pt files found in learning_cache!")

print(f"Found {len(pt_files)} cached samples")


# Load all samples (lazy: only metadata, not content)
def load_sample(idx):
    path = pt_files[idx]
    data = torch.load(path)
    # Data format: (input_tensor, label_tensor)
    x, y = data
    # x shape: [3, H, W]
    # y shape: [1]
    return x.numpy(), int(y.item())


# --- matplotlib setup ---
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
ax_map, ax_scan, ax_particles, ax_combined = axes

plt.subplots_adjust(bottom=0.25)  # space for slider


# Display function
def show_sample(idx):
    x, label = load_sample(idx)

    map_img = x[0]
    scan_img = x[1]
    particle_img = x[2]

    # Combined visualization (normalize to 0..1)
    combined = np.stack([map_img, scan_img, particle_img], axis=-1)
    combined = (combined - combined.min()) / (combined.max() + 1e-6)

    # Clear axes
    ax_map.clear()
    ax_scan.clear()
    ax_particles.clear()
    ax_combined.clear()

    ax_map.imshow(map_img, cmap="gray")
    ax_map.set_title("Channel 0: Map")

    ax_scan.imshow(scan_img, cmap="gray")
    ax_scan.set_title("Channel 1: Scan")

    ax_particles.imshow(particle_img, cmap="gray")
    ax_particles.set_title("Channel 2: Particle Cloud")

    ax_combined.imshow(combined)
    ax_combined.set_title(
        f"Combined Image\nLabel: {'DELOCALIZED' if label == 1 else 'LOCALIZED'}",
    )

    for ax in axes:
        ax.axis("off")

    fig.canvas.draw_idle()


# Slider axis
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
sample_slider = Slider(
    ax=ax_slider,
    label="Sample Index",
    valmin=0,
    valmax=len(pt_files) - 1,
    valinit=0,
    valstep=1,
)


# On slider change
def update(val):
    show_sample(int(sample_slider.val))


sample_slider.on_changed(update)

# Show first sample
show_sample(0)

plt.show()
