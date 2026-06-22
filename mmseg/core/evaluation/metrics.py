import mmcv
import numpy as np
import os

# def intersect_and_union(pred_label,
#                         label,
#                         num_classes,
#                         ignore_index,
#                         label_map=dict(),
#                         reduce_zero_label=False):
#     """Calculate intersection and Union.
#
#     Args:
#         pred_label (ndarray): Prediction segmentation map.
#         label (ndarray): Ground truth segmentation map.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         label_map (dict): Mapping old labels to new labels. The parameter will
#             work only when label is str. Default: dict().
#         reduce_zero_label (bool): Wether ignore zero label. The parameter will
#             work only when label is str. Default: False.
#
#      Returns:
#          ndarray: The intersection of prediction and ground truth histogram
#              on all classes.
#          ndarray: The union of prediction and ground truth histogram on all
#              classes.
#          ndarray: The prediction histogram on all classes.
#          ndarray: The ground truth histogram on all classes.
#     """
#
#     if isinstance(pred_label, str):
#         pred_label = np.load(pred_label)
#
#     if isinstance(label, str):
#         label = mmcv.imread(label, flag='unchanged', backend='pillow')
#     # modify if custom classes
#     if label_map is not None:
#         for old_id, new_id in label_map.items():
#             label[label == old_id] = new_id
#     if reduce_zero_label:
#         # avoid using underflow conversion
#         label[label == 0] = 255
#         label = label - 1
#         label[label == 254] = 255
#
#     mask = (label != ignore_index)
#     pred_label = pred_label[mask]
#     label = label[mask]
#
#     intersect = pred_label[pred_label == label]
#     area_intersect, _ = np.histogram(
#         intersect, bins=np.arange(num_classes + 1))
#     area_pred_label, _ = np.histogram(
#         pred_label, bins=np.arange(num_classes + 1))
#     area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
#     area_union = area_pred_label + area_label - area_intersect
#
#     return area_intersect, area_union, area_pred_label, area_label
def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and union."""

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    # 🧠 新增：将 label 的 255 转为 1（适配二分类标签）
    if label.max() == 255 and num_classes == 2:
        label = (label == 255).astype(np.uint8)  # 0: 背景，1: 前景

    # 可选的 label map 映射（如 COCO 风格）
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=float)
    total_area_label = np.zeros((num_classes, ), dtype=float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice
    
def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=None,
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics"""

    # 确保 gt_seg_maps 是 NumPy 数组
    gt_seg_maps = np.asarray(gt_seg_maps)

    # 转换标签中255为1，适用于二分类任务
    for i in range(len(gt_seg_maps)):
        if np.max(gt_seg_maps[i]) == 255:
            gt_seg_maps[i][gt_seg_maps[i] == 255] = 1  # 将255转为前景1

    if isinstance(metrics, str):
        metrics = [metrics]

    # ✅ 增加 mFscore
    allowed_metrics = ['mIoU', 'mDice', 'mRecall', 'mPrecision', 'mFscore', 'aAcc']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    # 计算所有必要的中间量
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = \
        total_intersect_and_union(results, gt_seg_maps,
                                  num_classes, ignore_index,
                                  label_map, reduce_zero_label)

    # 全局精度
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / (total_area_label + 1e-10)

    # 初始化返回指标
    ret_metrics = {
        'aAcc': all_acc,
        'Acc': acc,
    }

    # IoU
    if 'mIoU' in metrics:
        iou = total_area_intersect / (total_area_union + 1e-10)
        ret_metrics['mIoU'] = iou

    # Dice
    if 'mDice' in metrics:
        dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label + 1e-10)
        ret_metrics['mDice'] = dice

    # Recall
    recall = total_area_intersect / (total_area_label + 1e-10)
    precision = total_area_intersect / (total_area_pred_label + 1e-10)

    if 'mRecall' in metrics:
        ret_metrics['mRecall'] = recall

    if 'mPrecision' in metrics:
        ret_metrics['mPrecision'] = precision

    # ✅ F-score
    if 'mFscore' in metrics:
        fscore = 2 * precision * recall / (precision + recall + 1e-10)
        ret_metrics['mFscore'] = fscore

    # 处理 NaN
    if nan_to_num is not None:
        for k in ret_metrics:
            ret_metrics[k] = np.nan_to_num(ret_metrics[k], nan=nan_to_num)

    return ret_metrics
#-----------无Fscore-------------------
# def eval_metrics(results,
#                  gt_seg_maps,
#                  num_classes,
#                  ignore_index,
#                  metrics=None,
#                  nan_to_num=None,
#                  label_map=dict(),
#                  reduce_zero_label=False):
#     """Calculate evaluation metrics"""
#     # """Calculate evaluation metrics"""
#     # print("[评估调试] 预测结果类别分布:", np.unique(results[0]))
#     # print("[评估调试] 真实标签类别分布:", np.unique(gt_seg_maps[0]))
#     #
#     # # 确保输入是文件路径时检查文件是否存在
#     # if isinstance(results[0], str):
#     #     print(f"加载预测结果文件: {results[0]}")
#     #     if not os.path.exists(results[0]):
#     #         raise FileNotFoundError(f"预测结果文件不存在: {results[0]}")
#     #     results[0] = np.load(results[0])  # 加载 .npy 文件
#     #
#     # if isinstance(gt_seg_maps[0], str):
#     #     print(f"加载真实标签文件: {gt_seg_maps[0]}")
#     #     if not os.path.exists(gt_seg_maps[0]):
#     #         raise FileNotFoundError(f"真实标签文件不存在: {gt_seg_maps[0]}")
#     #     gt_seg_maps[0] = mmcv.imread(gt_seg_maps[0], flag='unchanged', backend='pillow')
#     # 将 255 转为 1，确保二值化的处理
#     # 转换为二值化标签，确保只有前景和背景
#     """Calculate evaluation metrics"""
#     # 确保 gt_seg_maps 是 NumPy 数组
#     gt_seg_maps = np.asarray(gt_seg_maps)

#     # 逐个处理每个图像标签，确保标签值为 0 或 1
#     for i in range(len(gt_seg_maps)):
#         # 如果标签图像中有 255，则转换为前景 1
#         if np.max(gt_seg_maps[i]) == 255:
#             gt_seg_maps[i][gt_seg_maps[i] == 255] = 1  # 将255转为前景1
#     if isinstance(metrics, str):
#         metrics = [metrics]

#     allowed_metrics = ['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc']
#     if not set(metrics).issubset(set(allowed_metrics)):
#         raise KeyError(f'metrics {metrics} is not supported')

#     total_area_intersect, total_area_union, total_area_pred_label, total_area_label = \
#         total_intersect_and_union(results, gt_seg_maps,
#                                   num_classes, ignore_index,
#                                   label_map, reduce_zero_label)

#     all_acc = total_area_intersect.sum() / total_area_label.sum()
#     acc = total_area_intersect / (total_area_label + 1e-10)  # avoid div0

#     ret_metrics = {
#         'aAcc': all_acc,
#         'Acc': acc,
#     }

#     if 'mIoU' in metrics:
#         iou = total_area_intersect / (total_area_union + 1e-10)
#         ret_metrics['mIoU'] = iou
#     if 'mDice' in metrics:
#         dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label + 1e-10)
#         ret_metrics['mDice'] = dice
#     if 'mRecall' in metrics:
#         recall = total_area_intersect / (total_area_label + 1e-10)
#         ret_metrics['mRecall'] = recall
#     if 'mPrecision' in metrics:
#         precision = total_area_intersect / (total_area_pred_label + 1e-10)
#         ret_metrics['mPrecision'] = precision

#     if nan_to_num is not None:
#         for k in ret_metrics:
#             ret_metrics[k] = np.nan_to_num(ret_metrics[k], nan=nan_to_num)

#     return ret_metrics
# def eval_metrics(results,
#                  gt_seg_maps,
#                  num_classes,
#                  ignore_index,
#                  metrics=None,
#                  nan_to_num=None,
#                  label_map=dict(),
#                  reduce_zero_label=False):
#     """Calculate evaluation metrics"""
#     print("[评估调试] 预测结果类别分布:", np.unique(results[0]))
#     print("[评估调试] 真实标签类别分布:", np.unique(gt_seg_maps[0]))
#     # gt_seg_maps=np.where(gt_seg_maps==255,1,gt_seg_maps)
#     # print("num_classes:", num_classes, "ignore_index:", ignore_index)
#     if isinstance(metrics, str):
#         metrics = [metrics]
#
#     allowed_metrics = ['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc']
#     if not set(metrics).issubset(set(allowed_metrics)):
#         raise KeyError(f'metrics {metrics} is not supported')
#
#     total_area_intersect, total_area_union, total_area_pred_label, total_area_label = \
#         total_intersect_and_union(results, gt_seg_maps,
#                                   num_classes, ignore_index,
#                                   label_map, reduce_zero_label)
#
#     all_acc = total_area_intersect.sum() / total_area_label.sum()
#     acc = total_area_intersect / (total_area_label + 1e-10)  # avoid div0
#
#     ret_metrics = {
#         'aAcc': all_acc,
#         'Acc': acc,
#     }
#
#     if 'mIoU' in metrics:
#         iou = total_area_intersect / (total_area_union + 1e-10)
#         ret_metrics['mIoU'] = iou
#     if 'mDice' in metrics:
#         dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label + 1e-10)
#         ret_metrics['mDice'] = dice
#     if 'mRecall' in metrics:
#         recall = total_area_intersect / (total_area_label + 1e-10)
#         ret_metrics['mRecall'] = recall
#     if 'mPrecision' in metrics:
#         precision = total_area_intersect / (total_area_pred_label + 1e-10)
#         ret_metrics['mPrecision'] = precision
#
#     if nan_to_num is not None:
#         for k in ret_metrics:
#             ret_metrics[k] = np.nan_to_num(ret_metrics[k], nan=nan_to_num)
#
#     return ret_metrics
