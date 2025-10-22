#!/usr/bin/env python3
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def enhance_particles(particle_img, amp=20, dilate=2, blur=True):
    img = particle_img.copy().astype(np.float32)
    if img.max() > 0:
        img /= img.max()
    img *= amp
    img = np.clip(img, 0, 1)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--amp", type=float, default=20)
    args = ap.parse_args()

    img, label = torch.load(args.path, map_location="cpu")
    img_np = img.detach().cpu().numpy()
    map_img, scan_img, particle_img = img_np

    variants = [
        ("original", particle_img),
        (
            "amp",
            enhance_particles(
                particle_img, amp=args.amp, dilate=0, blur=False
            ),
        ),
        (
            "amp+dilate",
            enhance_particles(
                particle_img, amp=args.amp, dilate=3, blur=False
            ),
        ),
        (
            "amp+dilate+blur",
            enhance_particles(particle_img, amp=args.amp, dilate=3, blur=True),
        ),
    ]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, part_img) in zip(axs, variants, strict=False):
        rgb = np.stack([map_img, scan_img, part_img], axis=-1)
        rgb = np.clip(rgb / (rgb.max() or 1), 0, 1)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
