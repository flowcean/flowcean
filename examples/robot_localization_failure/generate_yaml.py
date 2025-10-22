import os

base = "recordings/wgtl_half_split"

with open("training.yaml", "w") as f:
    f.write("rosbag:\n")
    f.write("  training_paths:\n")
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            f.write(f"    - {path}\n")

