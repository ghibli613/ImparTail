# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torchvision.ops import nms as tv_nms


def nms(boxes, scores, iou_threshold):
	return tv_nms(boxes, scores, iou_threshold)
