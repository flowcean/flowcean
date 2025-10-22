#!/usr/bin/env python3
import argparse

import torch

from flowcean.core.model import Model

# usage: uv run predict_single_tensor.py --model models/robot_localization.fml --tensor evaluation_cache/23.pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .fml model file")
    ap.add_argument(
        "--tensor",
        required=True,
        help="Path to one .pt tensor file",
    )
    args = ap.parse_args()

    # 1️⃣ Load model (.fml)
    wrapper = Model.load(args.model)
    print(f"Loaded model type: {type(wrapper).__name__}")

    # 2️⃣ Extract underlying CNN (module)
    model = getattr(wrapper, "module", None)
    if model is None:
        model = getattr(wrapper, "model", None)
    if model is None:
        raise SystemExit("❌ Could not find CNN module inside model wrapper")

    print(f"Using CNN class: {type(model).__name__}")

    # 3️⃣ Load tensor
    img, label = torch.load(args.tensor, map_location="cpu")
    x = img.unsqueeze(0)  # add batch dimension [1,3,H,W]

    # 4️⃣ Predict
    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    print(f"\nTensor: {args.tensor}")
    if label is not None:
        val = float(label.item() if label.numel() == 1 else label[0].item())
        print(
            f"Ground truth label: {val:.1f} "
            f"({'delocalized' if val >= 0.5 else 'localized'})",
        )

    print(f"Predicted probability (delocalized): {prob:.4f}")
    print(f"Predicted class: {'delocalized' if prob >= 0.5 else 'localized'}")


if __name__ == "__main__":
    main()
