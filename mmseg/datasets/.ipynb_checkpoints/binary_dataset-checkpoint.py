# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class BinarySegDataset(CustomDataset):
    CLASSES = ('background','samll_water')
    PALETTE = [[0,0,0],[255,255,255]]

    def __init__(self, **kwargs):
        super(BinarySegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=10,
            classes = ('background','samll_water'),
            palette = [[0,0,0],[255,255,255]],
            **kwargs)
        assert osp.exists(self.img_dir)

