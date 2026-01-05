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
        raise FileNotFoundError(f"无法读取图像: {path}")
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
    绘制输入、目标、各方法输出以及差异图的对比图。

    参数:
        gt_path (str): 目标（真实）图像路径。
        input_mr_path (str): 输入 MR 图像路径。
        prediction_items (Iterable): 可迭代对象，元素可以是 (name, path) 或仅包含 path。
        output_path (str): 最终对比图的保存路径。
        output_heatmap_dir (str | None): 若提供则保存各差异图到该目录。
        resize (tuple | None): 统一的图像尺寸，None 表示保持原尺寸。
        roi (tuple | None): (x1, y1, x2, y2)，用于放大显示的 ROI；传 None 关闭 ROI 行。
        zoom_factor (int): ROI 放大倍数。
        diff_cmap (str): 差异图配色，'jet' 等彩色或 'gray' 黑白。
        window_width_hu (float): 目标图窗宽（HU），用于推断差异图的 HU 刻度。
        window_level_hu (float): 目标图窗位（HU），用于刻度说明。
        figsize (tuple | None): Matplotlib 画布尺寸，None 时按方法数量自适应。
    """
    plt.close("all")
    gc.collect()

    gt_array = _load_grayscale(gt_path, resize)
    input_array = _load_grayscale(input_mr_path, resize)

    predictions = _normalize_predictions(prediction_items)
    if len(predictions) == 0:
        raise ValueError("prediction_items 不能为空")

    pred_arrays = []
    heatmaps_hu = []

    if output_heatmap_dir:
        os.makedirs(output_heatmap_dir, exist_ok=True)

    for idx, (name, path) in enumerate(predictions):
        pred_array = _load_grayscale(path, resize)
        if gt_array.shape != pred_array.shape:
            raise ValueError(f"预测图像 {path} 与目标图尺寸不匹配: {gt_array.shape} vs {pred_array.shape}")

        difference = np.abs(gt_array.astype(np.int16) - pred_array.astype(np.int16))
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

    # 第一行：输入、目标、各方法输出
    axes[0, 0].imshow(input_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input MR", fontsize=20)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Ground Truth CT", fontsize=20)
    axes[0, 1].axis("off")

    for i, (name, _) in enumerate(predictions):
        ax_pred = axes[0, i + 2]
        ax_pred.imshow(pred_arrays[i], cmap="gray", vmin=0, vmax=255)
        ax_pred.set_title(name, fontsize=16)
        ax_pred.axis("off")

    # ROI 行
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

    # 差异图行
    max_hu = np.max([heatmap.max() for heatmap in heatmaps_hu]) if heatmaps_hu else 1.0
    heatmap_axes = []
    im = None
    for i, (name, _) in enumerate(predictions):
        im = axes[2, i + 2].imshow(heatmaps_hu[i], cmap=diff_cmap, vmin=0, vmax=max_hu)
        axes[2, i + 2].set_title(f"{name} Δ", fontsize=16)
        axes[2, i + 2].axis("off")
        heatmap_axes.append(axes[2, i + 2])

    axes[2, 0].axis("off")
    axes[2, 1].axis("off")

    if heatmap_axes and im is not None:
        cbar = fig.colorbar(im, ax=heatmap_axes, orientation="horizontal", fraction=0.05, pad=0.1)
        cbar.set_label(f"Difference (HU)  |  WW={window_width_hu}, WL={window_level_hu}", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 示例：自行将路径替换为实际文件后运行
    example_gt = "ground_truth.jpg"
    example_input = "input_mr.jpg"
    example_predictions = [
        ("Method A", "pred1.png"),
        ("Method B", "pred2.png"),
    ]
    if all(os.path.exists(p) for p in [example_gt, example_input]) and all(os.path.exists(p) for _, p in example_predictions):
        create_comparison_figure(
            example_gt,
            example_input,
            example_predictions,
            output_path="comparison_example.pdf",
            output_heatmap_dir=None,
            diff_cmap="jet",
            window_width_hu=1500,
            window_level_hu=0,
        )
