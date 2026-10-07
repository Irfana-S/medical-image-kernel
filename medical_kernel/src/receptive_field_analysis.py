import os
import matplotlib.pyplot as plt


# ============================================================
# RESULTS FOLDER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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
# CONFIGURATIONS
# ============================================================

models = [
    ("3x3", 1, 3, "Fine/local details"),
    ("5x5", 1, 5, "Broader local structures"),
    ("3x3", 2, 5, "Broader context"),
    ("3x3", 3, 7, "Largest/broadest context")
]


# ============================================================
# PRINT TABLE
# ============================================================

print()
print("============================================================")
print("KERNEL SIZE AND DILATION ANALYSIS")
print("============================================================")

print(
    "{:<12} {:<10} {:<18} {:<25}".format(
        "Kernel",
        "Dilation",
        "Receptive Field",
        "Main Response"
    )
)

print("------------------------------------------------------------")

for kernel, dilation, rf, response in models:

    print(
        "{:<12} {:<10} {:<18} {:<25}".format(
            kernel,
            dilation,
            str(rf) + "x" + str(rf),
            response
        )
    )

print("============================================================")


# ============================================================
# CREATE VISUAL COMPARISON
# ============================================================

labels = [
    "3x3\nD=1",
    "5x5\nD=1",
    "3x3\nD=2",
    "3x3\nD=3"
]

receptive_fields = [3, 5, 5, 7]

plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    labels,
    receptive_fields
)

plt.xlabel(
    "Kernel Size and Dilation"
)

plt.ylabel(
    "Effective Receptive Field Size"
)

plt.title(
    "Receptive Field Comparison"
)

plt.ylim(
    0,
    8
)

for bar, value in zip(
    bars,
    receptive_fields
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        str(value) + "x" + str(value),
        ha="center"
    )

plt.tight_layout()


output_path = os.path.join(
    RESULTS_DIR,
    "receptive_field_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Saved:",
    output_path
)