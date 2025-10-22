import matplotlib.pyplot as plt
import torch

img, label = torch.load("learning_cache/3094.pt", map_location="cpu")
plt.imshow(img.permute(1, 2, 0))  # convert (C,H,W) → (H,W,C)
plt.title(f"Label: {label.item()}")
plt.axis("off")
plt.show()
