import matplotlib.pyplot as plt

from fastfuels_core.itd.local_maxima_filter import (
    variable_window_filter,
    fixed_window_filter,
)
from tests.itd.test_local_maxima_filter import (
    generate_complex_synthetic_chm,
    generate_mixed_morphology_chm,
)


def get_extent(chm_da):
    """Helper to calculate real-world extent for matplotlib's imshow."""
    transform = chm_da.rio.transform()
    width, height = chm_da.shape[1], chm_da.shape[0]
    min_x = transform.c
    max_y = transform.f
    max_x = min_x + (width * transform.a)
    min_y = max_y + (height * transform.e)
    return [min_x, max_x, min_y, max_y]


def visualize_algorithms():
    """Generates and displays all algorithm comparison visualizations."""

    # ==========================================
    # SCENARIO 1: COMPLEX FOREST (VWF vs FW)
    # ==========================================
    print("🌲 Generating Complex Synthetic CHM...")
    chm_da = generate_complex_synthetic_chm()
    ground_truth_complex = chm_da.attrs["ground_truth"]
    extent_complex = get_extent(chm_da)

    print("🔍 Running Filters on Complex Forest...")
    vwf_df = variable_window_filter(
        chm_da,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    ).compute()
    vwf_trees = vwf_df[vwf_df["height"] > 5.0]

    fw_df = fixed_window_filter(
        chm_da, min_height=2.0, spatial_resolution=0.5, window_size_meters=7.0
    ).compute()
    fw_trees = fw_df[fw_df["height"] > 5.0]

    gt_x_complex = [t["x"] for t in ground_truth_complex]
    gt_y_complex = [t["y"] for t in ground_truth_complex]

    # Plot Setup: Complex Forest
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig1.canvas.manager.set_window_title("Complex Forest Comparison")
    fig1.suptitle("Tree Detection Algorithm Comparison", fontsize=16, fontweight="bold")

    # Plot 1: VWF
    ax1.imshow(chm_da.values, extent=extent_complex, cmap="viridis", origin="upper")
    ax1.scatter(
        gt_x_complex,
        gt_y_complex,
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
        f"Variable Window Filter\nDetected: {len(vwf_trees)} / {len(ground_truth_complex)}",
        fontsize=12,
    )
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Easting (X)")
    ax1.set_ylabel("Northing (Y)")

    # Plot 2: FW
    ax2.imshow(chm_da.values, extent=extent_complex, cmap="viridis", origin="upper")
    ax2.scatter(
        gt_x_complex,
        gt_y_complex,
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
        f"Fixed Window Filter (7m Diameter)\nDetected: {len(fw_trees)} / {len(ground_truth_complex)}",
        fontsize=12,
    )
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Easting (X)")

    fig1.tight_layout()

    # ==========================================
    # SCENARIO 2: MIXED MORPHOLOGY (Irregular Canopies)
    # ==========================================
    print("🌲 Generating Mixed Morphology CHM (Conical + L-Shapes)...")
    mix_chm_da = generate_mixed_morphology_chm()
    ground_truth_mixed = mix_chm_da.attrs["ground_truth"]
    extent_mixed = get_extent(mix_chm_da)

    print("🔍 Running VWF on Mixed Forest...")
    vwf_mixed_df = variable_window_filter(
        mix_chm_da,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    ).compute()

    # Plot Setup: Mixed Morphology
    fig2, ax3 = plt.subplots(figsize=(10, 10))
    fig2.canvas.manager.set_window_title("Mixed Morphology Test")
    fig2.suptitle(
        "Robustness Check: Conical vs. Irregular Canopies",
        fontsize=16,
        fontweight="bold",
    )

    ax3.imshow(mix_chm_da.values, extent=extent_mixed, cmap="viridis", origin="upper")

    # Calculate old bug locations for L-shapes
    l_shape_gt = [t for t in ground_truth_mixed if t["type"] == "l_shape"]
    transform_mixed = mix_chm_da.rio.transform()
    bug_x = [
        transform_mixed.c + ((t["col"] + 5) * transform_mixed.a) for t in l_shape_gt
    ]
    bug_y = [
        transform_mixed.f + ((t["row"] + 6) * transform_mixed.e) for t in l_shape_gt
    ]

    ax3.scatter(
        bug_x,
        bug_y,
        facecolors="none",
        edgecolors="orange",
        s=200,
        linewidths=2,
        linestyle="--",
        label="Old Bug (Bounding Box Center)",
    )
    ax3.scatter(
        vwf_mixed_df["x"],
        vwf_mixed_df["y"],
        c="red",
        marker="x",
        s=80,
        linewidths=2,
        label="VWF Detected (maximum_position)",
    )

    ax3.set_title(
        "Notice the orange circles falling in the empty air of the 'L' canopies.\nOur red 'X' detections stay safely planted on the canopy pixels.",
        fontsize=11,
    )
    ax3.legend(loc="lower right")
    ax3.set_xlabel("Easting (X)")
    ax3.set_ylabel("Northing (Y)")

    fig2.tight_layout()

    # ==========================================
    # DISPLAY ALL FIGURES
    # ==========================================
    print("📊 Opening plot windows...")
    plt.show()


if __name__ == "__main__":
    visualize_algorithms()
