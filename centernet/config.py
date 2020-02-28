# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.DECONV_CHANNEL = [2048, 256, 128, 64]
    _C.MODEL.CENTERNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.MODULATE_DEFORM = True
    _C.MODEL.CENTERNET.BIAS_VALUE = -2.19
    _C.MODEL.CENTERNET.DOWN_SCALE = 4
    _C.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    _C.MODEL.CENTERNET.TENSOR_DIM = 128