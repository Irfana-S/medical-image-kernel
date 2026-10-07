import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "test",
    "images"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128


# ============================================================
# FOUR CONVOLUTION CONFIGURATIONS
# ============================================================

configs = [
    ("3x3_D1", 3, 1),
    ("5x5_D1", 5, 1),
    ("3x3_D2", 3, 2),
    ("3x3_D3", 3, 3)
]


# ============================================================
# LOAD ONE MEDICAL IMAGE
# ============================================================

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".bmp", ".png", ".jpg", ".jpeg")
    )
])

if len(files) == 0:
    raise RuntimeError(
        "No medical images found."
    )

image_name = files[0]

image_path = os.path.join(
    IMAGE_DIR,
    image_name
)

image = Image.open(
    image_path
).convert("L")

image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)

image_array = np.array(
    image
).astype(
    np.float32
) / 255.0

image_tensor = torch.tensor(
    image_array,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)


print("Medical image:", image_name)


# ============================================================
# CREATE FEATURE MAPS
# ============================================================

feature_maps = []

for name, kernel_size, dilation in configs:

    padding = (
        dilation
        * (kernel_size - 1)
        // 2
    )

    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        bias=False
    )

    # Fixed averaging kernel
    # This makes the comparison deterministic.
    with torch.no_grad():

        conv.weight.fill_(
            1.0 / (
                kernel_size
                * kernel_size
            )
        )

    output = conv(
        image_tensor
    )

    feature_map = output[
        0, 0
    ].detach().numpy()

    feature_maps.append(
        (name, feature_map)
    )


# ============================================================
# DISPLAY AND SAVE RESULTS
# ============================================================

plt.figure(
    figsize=(12, 8)
)

# Original image
plt.subplot(
    2, 3, 1
)

plt.imshow(
    image_array,
    cmap="gray"
)

plt.title(
    "Original Ultrasound"
)

plt.axis("off")


# Feature maps
for i, (name, feature_map) in enumerate(
    feature_maps
):

    plt.subplot(
        2, 3, i + 2
    )

    plt.imshow(
        feature_map,
        cmap="gray"
    )

    plt.title(
        name
    )

    plt.axis("off")


plt.tight_layout()


output_path = os.path.join(
    RESULTS_DIR,
    "feature_map_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print(
    "Feature-map comparison saved:"
)

print(
    output_path
)


# ============================================================
# RECEPTIVE FIELD TABLE
# ============================================================

print()
print("======================================")
print("RECEPTIVE FIELD ANALYSIS")
print("======================================")

for name, kernel_size, dilation in configs:

    receptive_field = (
        (kernel_size - 1)
        * dilation
        + 1
    )

    print(
        "{} | Kernel: {}x{} | Dilation: {} | RF: {}x{}".format(
            name,
            kernel_size,
            kernel_size,
            dilation,
            receptive_field,
            receptive_field
        )
    )

print("======================================")