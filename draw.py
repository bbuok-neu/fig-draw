import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
from typing import Iterable, Tuple, Union


MAX_PIXEL_VALUE = 255.0


def _load_grayscale(path, size=None):
    """Load a grayscale image and resize if needed."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if size is not None:
        image = cv2.resize(image, size)
    return np.array(image)


def _normalize_predictions(prediction_items):
    """Return a list of (name, path) pairs."""
    normalized = []
    for idx, item in enumerate(prediction_items):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((str(item[0]), item[1]))
        else:
            normalized.append((f"Pred {idx + 1}", item))
    return normalized


def create_comparison_figure(
    gt_path,
    input_mr_path,
    prediction_items: Iterable[Union[str, Tuple[str, str]]],
    *,
    output_path="comparison.pdf",
    output_heatmap_dir=None,
    resize=(256, 256),
    roi=(20, 80, 70, 130),
    zoom_factor=3,
    diff_cmap="jet",
    window_width_hu=1500.0,
    window_level_hu=0.0,
    figsize=None,
):
    """
    Build a comparison figure with input image, target image, multiple method outputs,
    and their difference maps.

    Args:
        gt_path (str): Path to the target (ground-truth) image.
        input_mr_path (str): Path to the input MR image.
        prediction_items (Iterable): Items may be (name, path) tuples or plain paths.
        output_path (str): Where to save the final comparison figure.
        output_heatmap_dir (str | None): If provided, saves per-method diff maps here.
        resize (tuple | None): Target size for all images; None keeps original size.
        roi (tuple | None): (x1, y1, x2, y2) ROI for zoomed row; None disables ROI row.
        zoom_factor (int): Zoom factor for ROI visualization.
        diff_cmap (str): Colormap for difference maps (e.g., 'jet' or 'gray').
        window_width_hu (float): Window width in HU, used for scaling pixel diffs to HU.
        window_level_hu (float): Window level in HU, shown on the colorbar label.
        figsize (tuple | None): Matplotlib figure size; None auto-scales by method count.

    Note:
        Assumes a linear mapping from pixel range 0-255 to HU using window_width_hu.
    """
    plt.close("all")
    gc.collect()

    gt_array = _load_grayscale(gt_path, resize)
    input_array = _load_grayscale(input_mr_path, resize)

    predictions = _normalize_predictions(prediction_items)
    if len(predictions) == 0:
        raise ValueError("prediction_items cannot be empty")

    pred_arrays = []
    heatmaps_hu = []

    if output_heatmap_dir:
        os.makedirs(output_heatmap_dir, exist_ok=True)

    for idx, (name, path) in enumerate(predictions):
        pred_array = _load_grayscale(path, resize)
        if gt_array.shape != pred_array.shape:
            raise ValueError(f"Prediction image {path} shape mismatch with target: {gt_array.shape} vs {pred_array.shape}")

        difference = np.abs(gt_array.astype(np.int16) - pred_array.astype(np.int16))
        # Assumes full 0-255 pixel range maps linearly to the provided HU window width.
        heatmap_hu = difference.astype(np.float32) * (window_width_hu / MAX_PIXEL_VALUE)

        pred_arrays.append(pred_array)
        heatmaps_hu.append(heatmap_hu)

        if output_heatmap_dir:
            heatmap_filename = os.path.join(output_heatmap_dir, f"heatmap_{idx + 1}.png")
            plt.imsave(heatmap_filename, heatmap_hu, cmap=diff_cmap)

    cols = 2 + len(pred_arrays)
    if figsize is None:
        figsize = (cols * 6, 18)

    fig, axes = plt.subplots(3, cols, figsize=figsize, squeeze=False)

    # Row 1: input, target, and method outputs
    axes[0, 0].imshow(input_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input MR", fontsize=30)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Ground Truth CT", fontsize=30)
    axes[0, 1].axis("off")

    for i, (name, _) in enumerate(predictions):
        ax_pred = axes[0, i + 2]
        ax_pred.imshow(pred_arrays[i], cmap="gray", vmin=0, vmax=255)
        ax_pred.set_title(name, fontsize=30)
        ax_pred.axis("off")

    # ROI row
    if roi is not None:
        x1, y1, x2, y2 = roi
        roi_input = input_array[y1:y2, x1:x2]
        roi_gt = gt_array[y1:y2, x1:x2]
        roi_preds = [pred[y1:y2, x1:x2] for pred in pred_arrays]

        roi_input_zoomed = cv2.resize(roi_input, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        roi_gt_zoomed = cv2.resize(roi_gt, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        roi_preds_zoomed = [
            cv2.resize(roi_pred, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
            for roi_pred in roi_preds
        ]

        axes[1, 0].imshow(roi_input_zoomed, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].axis("off")
        axes[1, 1].imshow(roi_gt_zoomed, cmap="gray", vmin=0, vmax=255)
        axes[1, 1].axis("off")

        for i, (name, _) in enumerate(predictions):
            axes[1, i + 2].imshow(roi_preds_zoomed[i], cmap="gray", vmin=0, vmax=255)
            axes[1, i + 2].axis("off")

        def _add_rect(target_ax):
            target_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lightcoral", facecolor="none"))

        _add_rect(axes[0, 0])
        _add_rect(axes[0, 1])
        for i in range(len(pred_arrays)):
            _add_rect(axes[0, i + 2])
    else:
        for ax in axes[1]:
            ax.axis("off")

    # Difference maps row
    max_hu = np.max([np.max(heatmap) for heatmap in heatmaps_hu]) if heatmaps_hu else 1.0
    heatmap_axes = []
    im = None
    for i, (name, _) in enumerate(predictions):
        im = axes[2, i + 2].imshow(heatmaps_hu[i], cmap=diff_cmap, vmin=0, vmax=max_hu)
        # axes[2, i + 2].set_title(f"{name} Δ", fontsize=16)
        axes[2, i + 2].axis("off")
        heatmap_axes.append(axes[2, i + 2])

    axes[2, 0].axis("off")
    axes[2, 1].axis("off")

    if heatmap_axes and im is not None:
        cbar = fig.colorbar(im, ax=heatmap_axes, orientation="horizontal", fraction=0.08, pad=0.1, cax=fig.add_axes([0.05, 0.05, 0.15, 0.015]))
        cbar.set_label(f"Difference (HU)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _example_files_exist(gt_path, input_path, prediction_items):
    normalized = _normalize_predictions(prediction_items)
    return all(os.path.exists(p) for p in [gt_path, input_path]) and all(os.path.exists(path) for _, path in normalized)

if __name__ == "__main__":
    # Example usage: replace the paths below with real files before running
    example_gt = "ground_truth.jpg"
    example_input = "input_mr.jpg"
    example_predictions = [
        ("EGSDE", "EGSDE.jpg"),
        ("MIDiffusion", "MIDiffusion.png"),
        ("FGDM", "FGDM.jpg"),
        ("SDEdit", "SDEdit.png"),
        ("StyleGAN", "StyleGAN.png"),
        ("ours", "ours.png"),
    ]
    if _example_files_exist(example_gt, example_input, example_predictions):
        create_comparison_figure(
            example_gt,
            example_input,
            example_predictions,
            output_path="comparison_example.pdf",
            output_heatmap_dir=None,
            diff_cmap="gray",
            window_width_hu=1500,
            window_level_hu=300,
            roi=(40,100,90,150)
        )
