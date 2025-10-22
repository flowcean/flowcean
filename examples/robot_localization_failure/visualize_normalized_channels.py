import matplotlib.pyplot as plt
import torch

# --- Load your tensor ---
path = "learning_cache/7728.pt"
img, label = torch.load(path, map_location="cpu")

print(f"\nLoaded: {path}")
print(f"Label: {label.item()}")
print(f"Original shape: {tuple(img.shape)}  dtype: {img.dtype}")

# --- Print stats before normalization ---
print("\nBefore normalization:")
for i, name in enumerate(["R", "G", "B"]):
    ch = img[i]
    print(
        f"{name}: mean={ch.mean():.5f}, std={ch.std():.5f}, min={ch.min():.3f}, max={ch.max():.3f}",
    )

# --- Normalize each channel individually to [0, 1] ---
img_norm = torch.zeros_like(img)
for c in range(3):
    ch = img[c]
    ch_min, ch_max = ch.min(), ch.max()
    if (ch_max - ch_min) > 1e-8:
        img_norm[c] = (ch - ch_min) / (ch_max - ch_min)
    else:
        img_norm[c] = ch  # if constant channel, leave it

# --- Print stats after normalization ---
print("\nAfter normalization:")
for i, name in enumerate(["R", "G", "B"]):
    ch = img_norm[i]
    print(
        f"{name}: mean={ch.mean():.5f}, std={ch.std():.5f}, min={ch.min():.3f}, max={ch.max():.3f}",
    )

# --- Visualization ---
fig, axs = plt.subplots(2, 4, figsize=(14, 7))

# Top row: before normalization
axs[0, 0].imshow(img.permute(1, 2, 0))
axs[0, 0].set_title("Combined RGB (before)")
for i, name in enumerate(["R", "G", "B"]):
    axs[0, i + 1].imshow(img[i], cmap="gray", vmin=0, vmax=1)
    axs[0, i + 1].set_title(f"{name} (before)")

# Bottom row: after normalization
axs[1, 0].imshow(img_norm.permute(1, 2, 0))
axs[1, 0].set_title("Combined RGB (after)")
for i, name in enumerate(["R", "G", "B"]):
    axs[1, i + 1].imshow(img_norm[i], cmap="gray", vmin=0, vmax=1)
    axs[1, i + 1].set_title(f"{name} (after)")

for ax in axs.flatten():
    ax.axis("off")

plt.suptitle(f"Visualization of {path} (label={label.item()})", fontsize=12)
plt.tight_layout()
plt.show()
