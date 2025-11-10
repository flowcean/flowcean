"""inspect_tensor_images.py

Quick visualization script for inspecting a single saved image tensor (.pt file).
It loads a (image, label) pair, converts the tensor from (C,H,W) to (H,W,C),
and displays the image with its corresponding label using Matplotlib.
"""

import matplotlib.pyplot as plt
import torch

img, label = torch.load(
    "../learning_cache/7.pt",
    map_location="cpu",
)
plt.imshow(
    img.permute(1, 2, 0)
)  # convert (C,H,W) → (H,W,C) - Matplotlib format
plt.title(f"Label: {label.item()}")
plt.axis("off")
plt.show()
