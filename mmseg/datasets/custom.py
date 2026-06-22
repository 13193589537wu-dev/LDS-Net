import os
import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette
    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate with metrics: mIoU, mDice, mRecall, mPrecision, mFscore, aAcc."""
        if isinstance(metric, str):
            metric = [metric]
    
        allowed_metrics = ['mIoU', 'mDice', 'mRecall', 'mPrecision', 'mFscore', 'aAcc']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError(f'metric {metric} is not supported')
    
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
    
        # 类别数量 & 名称
        if self.CLASSES is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
            class_names = tuple(range(num_classes))
        else:
            num_classes = len(self.CLASSES)
            class_names = self.CLASSES
    
        # 调用核心指标计算函数
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metrics=metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label
        )
    
        # 拿出各指标
        iou = ret_metrics.get('mIoU', [0] * num_classes)
        dice = ret_metrics.get('mDice', [0] * num_classes)
        recall = ret_metrics.get('mRecall', [0] * num_classes)
        precision = ret_metrics.get('mPrecision', [0] * num_classes)
        fscore = ret_metrics.get('mFscore', [0] * num_classes)
        acc = ret_metrics.get('Acc', [0] * num_classes)
        all_acc = ret_metrics.get('aAcc', 0)
    
        # ✅ 类别级表格（新增 Fscore）
        class_table_data = [['Class', 'IoU', 'Dice', 'Recall', 'Precision', 'Fscore', 'Acc']]
        for i in range(num_classes):
            class_table_data.append([
                class_names[i],
                round(iou[i] * 100, 2),
                round(dice[i] * 100, 2),
                round(recall[i] * 100, 2),
                round(precision[i] * 100, 2),
                round(fscore[i] * 100, 2),
                round(acc[i] * 100, 2)
            ])
    
        # ✅ 汇总表（新增 mFscore）
        summary_table_data = [['Scope', 'mIoU', 'mDice', 'mRecall', 'mPrecision', 'mFscore', 'aAcc']]
        summary_table_data.append([
            'global',
            round(np.nanmean(iou) * 100, 2),
            round(np.nanmean(dice) * 100, 2),
            round(np.nanmean(recall) * 100, 2),
            round(np.nanmean(precision) * 100, 2),
            round(np.nanmean(fscore) * 100, 2),
            round(all_acc * 100, 2)
        ])
    
        # 打印结果
        print_log('Per class results:', logger)
        print_log('\n' + AsciiTable(class_table_data).table, logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + AsciiTable(summary_table_data).table, logger=logger)
    
        # ✅ 保存 eval_results
        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mDice'] = np.nanmean(dice)
        eval_results['mRecall'] = np.nanmean(recall)
        eval_results['mPrecision'] = np.nanmean(precision)
        eval_results['mFscore'] = np.nanmean(fscore)
        eval_results['aAcc'] = all_acc
        #-------------------------------------显示各类指标---------------------------------------------
        for i in range(num_classes):
            per_cls_str = (f"{class_names[i]} -> IoU: {iou[i]*100:.2f}, Dice: {dice[i]*100:.2f}, "
                           f"Recall: {recall[i]*100:.2f}, Precision: {precision[i]*100:.2f}, "
                           f"Fscore: {fscore[i]*100:.2f}, Acc: {acc[i]*100:.2f}")
            print_log(per_cls_str, logger)
        #----------------------------------------------------------------------------------------------
        # 清理缓存文件
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
    
        return eval_results
    # def evaluate(self,
    #              results,
    #              metric=None,
    #              logger=None,
    #              efficient_test=False,
    #              **kwargs):
    #     """Evaluate with metrics: mIoU, mDice, mRecall, mPrecision, aAcc."""
    #     if isinstance(metric, str):
    #         metric = [metric]

    #     allowed_metrics = ['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError(f'metric {metric} is not supported')

    #     eval_results = {}
    #     gt_seg_maps = self.get_gt_seg_maps(efficient_test)

    #     if self.CLASSES is None:
    #         num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
    #         class_names = tuple(range(num_classes))
    #     else:
    #         num_classes = len(self.CLASSES)
    #         class_names = self.CLASSES

    #     ret_metrics = eval_metrics(
    #         results,
    #         gt_seg_maps,
    #         num_classes,
    #         self.ignore_index,
    #         metrics=metric,
    #         label_map=self.label_map,
    #         reduce_zero_label=self.reduce_zero_label
    #     )

    #     iou = ret_metrics.get('mIoU', [0] * num_classes)
    #     dice = ret_metrics.get('mDice', [0] * num_classes)
    #     recall = ret_metrics.get('mRecall', [0] * num_classes)
    #     precision = ret_metrics.get('mPrecision', [0] * num_classes)
    #     acc = ret_metrics.get('Acc', [0] * num_classes)
    #     all_acc = ret_metrics.get('aAcc', 0)

    #     class_table_data = [['Class', 'IoU', 'Dice', 'Recall', 'Precision', 'Acc']]
    #     for i in range(num_classes):
    #         class_table_data.append([
    #             class_names[i],
    #             round(iou[i] * 100, 2),
    #             round(dice[i] * 100, 2),
    #             round(recall[i] * 100, 2),
    #             round(precision[i] * 100, 2),
    #             round(acc[i] * 100, 2)
    #         ])

    #     summary_table_data = [['Scope', 'mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc']]
    #     summary_table_data.append([
    #         'global',
    #         round(np.nanmean(iou) * 100, 2),
    #         round(np.nanmean(dice) * 100, 2),
    #         round(np.nanmean(recall) * 100, 2),
    #         round(np.nanmean(precision) * 100, 2),
    #         round(all_acc * 100, 2)
    #     ])

    #     print_log('Per class results:', logger)
    #     print_log('\n' + AsciiTable(class_table_data).table, logger=logger)
    #     print_log('Summary:', logger)
    #     print_log('\n' + AsciiTable(summary_table_data).table, logger=logger)

    #     eval_results['mIoU'] = np.nanmean(iou)
    #     eval_results['mDice'] = np.nanmean(dice)
    #     eval_results['mRecall'] = np.nanmean(recall)
    #     eval_results['mPrecision'] = np.nanmean(precision)
    #     eval_results['aAcc'] = all_acc

    #     if mmcv.is_list_of(results, str):
    #         for file_name in results:
    #             os.remove(file_name)

    #     return eval_results
