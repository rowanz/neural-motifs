import torch
import numpy as np
from torch.nn import functional as F
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps as bbox_overlaps_np
from lib.fpn.box_intersections_cpu.bbox import bbox_intersections as bbox_intersections_np


def bbox_loss(prior_boxes, deltas, gt_boxes, eps=1e-4, scale_before=1):
    """
    Computes the loss for predicting the GT boxes from prior boxes
    :param prior_boxes: [num_boxes, 4] (x1, y1, x2, y2)
    :param deltas: [num_boxes, 4]    (tx, ty, th, tw)
    :param gt_boxes: [num_boxes, 4] (x1, y1, x2, y2)
    :return:
    """
    prior_centers = center_size(prior_boxes) #(cx, cy, w, h)
    gt_centers = center_size(gt_boxes) #(cx, cy, w, h)

    center_targets = (gt_centers[:, :2] - prior_centers[:, :2]) / prior_centers[:, 2:]
    size_targets = torch.log(gt_centers[:, 2:]) - torch.log(prior_centers[:, 2:])
    all_targets = torch.cat((center_targets, size_targets), 1)

    loss = F.smooth_l1_loss(deltas, all_targets, size_average=False)/(eps + prior_centers.size(0))

    return loss


def bbox_preds(boxes, deltas):
    """
    Converts "deltas" (predicted by the network) along with prior boxes
    into (x1, y1, x2, y2) representation.
    :param boxes: Prior boxes, represented as (x1, y1, x2, y2)
    :param deltas: Offsets (tx, ty, tw, th)
    :param box_strides [num_boxes,] distance apart between boxes. anchor box can't go more than
       \pm box_strides/2 from its current position. If None then we'll use the widths
       and heights
    :return: Transformed boxes
    """

    if boxes.size(0) == 0:
        return boxes
    prior_centers = center_size(boxes)

    xys = prior_centers[:, :2] + prior_centers[:, 2:] * deltas[:, :2]

    whs = torch.exp(deltas[:, 2:]) * prior_centers[:, 2:]

    return point_form(torch.cat((xys, whs), 1))


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0

    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] - 0.5 * boxes[:, 2:],
                                boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)))
    return torch.cat((boxes[:, :2] - 0.5 * boxes[:, 2:],
                      boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)), 1)  # xmax, ymax


###########################################################################
### Torch Utils, creds to Max de Groot
###########################################################################

def bbox_intersections(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_intersections_np(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_overlaps_np(box_a, box_b)

    inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
              (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
              (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:,:,:,0]*inter[:,:,:,1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:,2]- boxes_flat[:,0]+1.0)*(
        boxes_flat[:,3]- boxes_flat[:,1]+1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union

