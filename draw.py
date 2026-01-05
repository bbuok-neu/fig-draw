import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from matplotlib.patches import Rectangle

def calculate_and_display_heatmaps(gt_path, input_mr_path, prediction_paths, output_dir="heatmaps"):
    """
    计算并显示预测图像与真实值图像之间的差异热力图。

    参数:
        gt_path (str): 真实值图像的路径。
        input_mr_path (str): 输入MR图像的路径
        prediction_paths (list): 包含9个预测图像路径的列表。
        output_dir (str): 保存热力图的目录，默认为 "heatmaps"。
    """

    # 读取真实值图像
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
    gt_image = cv2.resize(gt_image, (256,256))
    gt_array = np.array(gt_image)

    # 读取输入MR图像
    input_image = cv2.imread(input_mr_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
    input_image = cv2.resize(input_image, (256, 256))
    input_array = np.array(input_image)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 存储热力图
    heatmaps = []

    # 循环处理每个预测图像
    for i, pred_path in enumerate(prediction_paths):
        # 读取预测图像
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
        pred_array = np.array(pred_image)

        # 确保图像大小相同
        if gt_array.shape != pred_array.shape:
            print(f"警告: 预测图像 {pred_path} 的大小与真实值图像不匹配。跳过此图像。")
            continue

        # 计算差值并取绝对值
        difference = np.abs((gt_array).astype(np.int16) - pred_array.astype(np.int16))
        difference = (difference).astype(np.uint8)

        # 归一化
        # difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))

        heatmaps.append(difference)

        # 保存热力图
        heatmap_filename = os.path.join(output_dir, f"heatmap_{i + 1}.png")
        plt.imsave(heatmap_filename, difference, cmap="jet")

        # 显示图像
        fig, axes = plt.subplots(2, 11, figsize=(44, 9))

        # 显示输入MR图像 (放大)
        axes[0][0].imshow(input_array, cmap="gray")
        axes[0][0].set_title("Input MR")
        axes[0][0].axis("off")
        axes[0][0].set_position([0.065, 0.55, 0.07, 0.4])

        # 显示真实值图像 (放大)
        axes[0][1].imshow(gt_array, cmap="gray")
        axes[0][1].set_title("Ground Truth")
        axes[0][1].axis("off")
        axes[0][1].set_position([0.14, 0.55, 0.07, 0.4])

        # 显示各个模型输出的图像
        for i in range(len(heatmaps)):
            axes[0][i + 2].imshow(cv2.imread(prediction_paths[i], cv2.IMREAD_GRAYSCALE), cmap='gray')
            axes[0][i + 2].set_title(f"Pred {i + 1}", fontsize=8)
            axes[0][i + 2].axis('off')
            axes[0][i + 2].set_position([0.23 + 0.075 * i, 0.55, 0.07, 0.4])

        # 显示热力图
        for i in range(len(heatmaps)):
            im = axes[1][i + 2].imshow(heatmaps[i], cmap="jet")
            axes[1][i + 2].set_title(f"Heatmap {i + 1}", fontsize=8)
            axes[1][i + 2].axis("off")
            axes[1][i + 2].set_position([0.23 + 0.075 * i, 0.05, 0.07, 0.4])  # 与Pred图x坐标相同

        # 隐藏多余的子图
        axes[0][-1].axis("off")
        axes[1][0].axis("off")
        axes[1][1].axis("off")

        # 添加一个共享的颜色条
        cax = fig.add_axes([0.23, 0.05, 0.67, 0.03])  # [左, 下, 宽, 高]
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)

        # plt.show()
        # plt.close('all')
        # gc.collect()


def calculate_and_display_heatmaps_2(gt_path, input_mr_path, prediction_paths, output_dir="heatmaps"):
    plt.close('all')
    gc.collect()

    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
    gt_image = cv2.resize(gt_image, (256, 256))
    gt_array = np.array(gt_image)

    # 读取输入MR图像
    input_image = cv2.imread(input_mr_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
    input_image = cv2.resize(input_image, (256, 256))
    input_array = np.array(input_image)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 存储热力图
    heatmaps = []
    heatmaps_raw = []

    # 循环处理每个预测图像
    for i, pred_path in enumerate(prediction_paths):
        # 读取预测图像
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像
        pred_array = np.array(pred_image)

        # 确保图像大小相同
        if gt_array.shape != pred_array.shape:
            print(f"警告: 预测图像 {pred_path} 的大小与真实值图像不匹配。跳过此图像。")
            continue

        # 计算差值并取绝对值
        difference = np.abs((gt_array).astype(np.int16) - pred_array.astype(np.int16))
        difference = (difference).astype(np.uint8)

        # 归一化
        # difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))

        heatmaps.append(difference)
        heatmaps_raw.append(np.round(difference.astype(np.float32) * 5.859375).astype(np.int32))
        # 保存热力图
        heatmap_filename = os.path.join(output_dir, f"heatmap_{i + 1}.png")
        plt.imsave(heatmap_filename, difference, cmap="jet")

    heatmaps_norm = []
    hmax = heatmaps[0].max()
    hmin = heatmaps[0].min()
    for i in range(len(heatmaps)):
        if heatmaps[i].max() > hmax:
            hmax = heatmaps[i].max()
        if heatmaps[i].min() < hmin:
            hmin = heatmaps[i].min()
    for i in range(len(heatmaps)):
        heatmaps_norm.append(255*((heatmaps[i] - hmin ) / (hmax - hmin)))


    name_list = ['CycleGAN','UNIT','CUT','Reg-GAN','AttentionGAN','SC-CycleGAN','DC-CycleGAN','UVCGAN', 'Pix2pix', 'Ours']

    x1, y1 = 20, 80
    x2, y2 = 70, 130

    # 放大倍数
    zoom_factor = 3


    fig = plt.figure(figsize=(72, 21))
    gs = plt.GridSpec(3, 12, width_ratios=[1] * 12, height_ratios=[1, 1, 1])

    # 显示输入MR图像
    ax_input = plt.subplot(gs[0, 0])
    ax_input.imshow(input_array, cmap="gray")
    ax_input.set_title("Input MR", fontsize=48)
    ax_input.axis("off")

    # 显示真实值图像
    ax_gt = plt.subplot(gs[0, 1])
    ax_gt.imshow(gt_array, cmap="gray")
    ax_gt.set_title("Ground Truth CT", fontsize=48)
    ax_gt.axis("off")

    # 显示各个模型输出的图像
    for i in range(len(heatmaps)):
        ax_pred = plt.subplot(gs[0, i + 2])
        ax_pred.imshow(cv2.imread(prediction_paths[i], cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax_pred.set_title(name_list[i], fontsize=48)
        ax_pred.axis('off')


    roi_input = input_array[y1:y2, x1:x2]
    roi_gt = gt_array[y1:y2, x1:x2]
    roi_heatmaps = [heatmap[y1:y2, x1:x2] for heatmap in heatmaps]
    roi_preds = [cv2.imread(prediction_paths[i], cv2.IMREAD_GRAYSCALE)[y1:y2, x1:x2] for i in range(len(heatmaps))]

    roi_input_zoomed = cv2.resize(roi_input, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    roi_gt_zoomed = cv2.resize(roi_gt, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    roi_heatmaps_zoomed = [cv2.resize(roi_heatmap, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
                           for roi_heatmap in roi_heatmaps]
    roi_preds_zoomed = [cv2.resize(roi_pred, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR) for
                        roi_pred in roi_preds]

    ax_roi_input = plt.subplot(gs[1, 0])  # 调换到第二行
    ax_roi_input.imshow(roi_input_zoomed, cmap="gray",vmin=0, vmax=255)
    # ax_roi_input.set_title("Input MR (Zoomed)", fontsize=8)
    ax_roi_input.axis("off")
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='lightcoral', facecolor='none')
    ax_input.add_patch(rect)

    ax_roi_gt = plt.subplot(gs[1, 1])  # 调换到第二行
    ax_roi_gt.imshow(roi_gt_zoomed, cmap="gray", vmin=0, vmax=255)
    # ax_roi_gt.set_title("GT CT (Zoomed)", fontsize=8)
    ax_roi_gt.axis("off")
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='lightcoral', facecolor='none')
    ax_gt.add_patch(rect)

    for i in range(len(heatmaps)):
        ax_roi_pred = plt.subplot(gs[1, i + 2])  # 调换到第二行
        ax_roi_pred.imshow(roi_preds_zoomed[i], cmap='gray',vmin=0, vmax=255)
        # ax_roi_pred.set_title(f"Pred {i + 1} (Zoomed)", fontsize=8)
        ax_roi_pred.axis("off")
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='lightcoral', facecolor='none')
        plt.subplot(gs[0, i + 2]).add_patch(rect)  # 在原图上添加roi框

    # 显示热力图
    for i in range(len(heatmaps)):
        ax_heatmap = plt.subplot(gs[2, i + 2])  # 调换到第三行
        im = ax_heatmap.imshow(heatmaps_norm[i], cmap="gray")
        # ax_heatmap.set_title(f"Heatmap {i + 1}", fontsize=8)
        ax_heatmap.axis("off")

    # 隐藏多余的子图
    for j in range(2):
        plt.subplot(gs[2, j]).axis("off")

    # 添加一个共享的颜色条
    colorbar_axes_position = [0.01, 0.07, 0.15, 0.015]  # [left, bottom, width, height]
    cax = fig.add_axes(colorbar_axes_position)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    ticks = cbar.get_ticks()

    # 计算缩放后的 HU 值刻度
    max_hu = (1500) / 255  # 假设所有热图的最大 HU 值相同
    hu_ticks = [round(tick * max_hu) for tick in ticks]

    # 设置刻度标签，并添加 "HU" 单位
    cbar.ax.set_xticklabels(['{:.0f}'.format(hu_tick) for hu_tick in hu_ticks])
    cbar.ax.tick_params(labelsize=36)

    plt.tight_layout()
    # plt.show()
    plt.savefig('output_8_24.pdf')


# 示例用法
gt_path = "ground_truth.jpg"  # 替换为真实值图像的路径 (JPG 或 PNG)
input_mr_path = "input_mr.jpg"  # 替换为输入MR图像的路径 (JPG 或 PNG)
prediction_paths = [
    "pred1.png",
    "pred2.jpg",
    "pred3.png",
    "pred4.jpg",
    "pred5.png",
    "pred6.jpg",
    "pred7.jpg",
    "pred8.png",
    "pred9.png",
    # "pred10.jpg",
    "generated_ct_34.png",
]

calculate_and_display_heatmaps_2(gt_path, input_mr_path, prediction_paths)
