import matplotlib.pyplot as plt

# Adjust these import paths based on your exact project structure
from fastfuels_core.itd.local_maxima_filter import (
    variable_window_filter,
    fixed_window_filter,
)
from tests.itd.test_local_maxima_filter import generate_complex_synthetic_chm


def visualize_algorithms():
    print("🌲 Generating synthetic CHM...")
    chm_da = generate_complex_synthetic_chm()
    ground_truth = chm_da.attrs["ground_truth"]

    print("🔍 Running Variable Window Filter...")
    vwf_df = variable_window_filter(
        chm_da,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    ).compute()
    vwf_trees = vwf_df[vwf_df["height"] > 5.0]

    print("🔍 Running Fixed Window Filter (7m)...")
    fw_df = fixed_window_filter(
        chm_da, min_height=2.0, spatial_resolution=0.5, window_size_meters=7.0
    ).compute()
    fw_trees = fw_df[fw_df["height"] > 5.0]

    # --- Plotting Setup ---
    gt_x = [t["x"] for t in ground_truth]
    gt_y = [t["y"] for t in ground_truth]

    # Calculate real-world extent for matplotlib's imshow
    transform = chm_da.rio.transform()
    width, height = chm_da.shape[1], chm_da.shape[0]
    min_x = transform.c
    max_y = transform.f
    max_x = min_x + (width * transform.a)
    min_y = max_y + (height * transform.e)  # e is usually negative
    extent = [min_x, max_x, min_y, max_y]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle("Tree Detection Algorithm Comparison", fontsize=16, fontweight="bold")

    # --- Plot 1: Variable Window Filter ---
    ax1.imshow(chm_da.values, extent=extent, cmap="viridis", origin="upper")
    ax1.scatter(
        gt_x,
        gt_y,
        facecolors="none",
        edgecolors="lime",
        s=150,
        linewidths=2,
        label="Ground Truth",
    )
    ax1.scatter(
        vwf_trees["x"], vwf_trees["y"], c="red", marker="x", s=60, label="VWF Detected"
    )

    ax1.set_title(
        f"Variable Window Filter\nDetected: {len(vwf_trees)} / {len(ground_truth)}",
        fontsize=12,
    )
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Easting (X)")
    ax1.set_ylabel("Northing (Y)")

    # --- Plot 2: Fixed Window Filter ---
    ax2.imshow(chm_da.values, extent=extent, cmap="viridis", origin="upper")
    ax2.scatter(
        gt_x,
        gt_y,
        facecolors="none",
        edgecolors="lime",
        s=150,
        linewidths=2,
        label="Ground Truth",
    )
    ax2.scatter(
        fw_trees["x"], fw_trees["y"], c="red", marker="x", s=60, label="FW Detected"
    )

    ax2.set_title(
        f"Fixed Window Filter (7m Diameter)\nDetected: {len(fw_trees)} / {len(ground_truth)}",
        fontsize=12,
    )
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Easting (X)")

    plt.tight_layout()
    print("📊 Opening plot window...")
    plt.show()


if __name__ == "__main__":
    visualize_algorithms()
