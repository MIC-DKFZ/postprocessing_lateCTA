# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from nndet.io.load import load_pickle
from nndet.core.boxes.ops_np import box_iou_np

file = "/media/E132-Projekte/Projects/2025_MartinezMora_TimeAwareVODetection/data_stroke_amsterdamproject/Task079_intracranial/preprocessed/labelsTs/ukb_0446_boxes_gt.npz"
arr = np.load(file)["boxes"]


file_pred = "/media/E132-Projekte/Projects/2025_MartinezMora_TimeAwareVODetection/models_stroke_amsterdamproject/Task079_intracranial/StrokeRetinaNetV001A2_NoCropBgPosPlannerA2Blosc_3d_nndetv2_phaseShift/fold0/test_predictions_original/ukb_0446_boxes.pkl"
pred = load_pickle(file_pred)
boxes = pred["pred_boxes"]
scores = pred["pred_scores"]

iou = box_iou_np(boxes, arr)
ind = np.where(iou > 0.10)[0]
print(scores[ind], boxes[ind], arr, iou.shape)
