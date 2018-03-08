# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time

def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box

    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:,0]]
    obj_scores1 = obj_scores.data[rel_inds[:,1]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:,1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

    rels = rel_inds[rel_scores_idx].cpu().numpy()
    pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()

    return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted

# def _get_similar_boxes(boxes, obj_classes_topk, nms_thresh=0.3):
#     """
#     Assuming bg is NOT A LABEL.
#     :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
#     :param obj_classes: [num_box, topk] class labels
#     :return: num_box, topk, num_box, topk array containing similarities.
#     """
#     topk = obj_classes_topk.size(1)
#     num_box = boxes.size(0)
#
#     box_flat = boxes.view(-1, 4) if boxes.dim() == 3 else boxes[:, None].expand(
#         num_box, topk, 4).contiguous().view(-1, 4)
#     jax = bbox_overlaps(box_flat, box_flat).data > nms_thresh
#     # Filter out things that are not gonna compete.
#     classes_eq = obj_classes_topk.data.view(-1)[:, None] == obj_classes_topk.data.view(-1)[None, :]
#     jax &= classes_eq
#     boxes_are_similar = jax.view(num_box, topk, num_box, topk)
#     return boxes_are_similar.cpu().numpy().astype(np.bool)
