"""
Generates anchor targets to train the detector. Does this during the collate step in training
as it's much cheaper to do this on a separate thread.

Heavily adapted from faster_rcnn/rpn_msr/anchor_target_layer.py.
"""
import numpy as np
import numpy.random as npr

from config import IM_SCALE, RPN_NEGATIVE_OVERLAP, RPN_POSITIVE_OVERLAP, \
    RPN_BATCHSIZE, RPN_FG_FRACTION, ANCHOR_SIZE, ANCHOR_SCALES, ANCHOR_RATIOS
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.fpn.generate_anchors import generate_anchors


def anchor_target_layer(gt_boxes, im_size, 
                        allowed_border=0):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.

    for each (H, W) location i
      generate 3 anchor boxes centered on cell i
    filter out-of-image anchors
    measure GT overlap

    :param gt_boxes: [x1, y1, x2, y2] boxes. These are assumed to be at the same scale as
                     the image (IM_SCALE)
    :param im_size: Size of the image (h, w). This is assumed to be scaled to IM_SCALE
    """
    if max(im_size) != IM_SCALE:
        raise ValueError("im size is {}".format(im_size))
    h, w = im_size

    # Get the indices of the anchors in the feature map.
    # h, w, A, 4
    ans_np = generate_anchors(base_size=ANCHOR_SIZE,
                              feat_stride=16,
                              anchor_scales=ANCHOR_SCALES,
                              anchor_ratios=ANCHOR_RATIOS,
                              )
    ans_np_flat = ans_np.reshape((-1, 4))
    inds_inside = np.where(
        (ans_np_flat[:, 0] >= -allowed_border) &
        (ans_np_flat[:, 1] >= -allowed_border) &
        (ans_np_flat[:, 2] < w + allowed_border) &  # width
        (ans_np_flat[:, 3] < h + allowed_border)  # height
    )[0]
    good_ans_flat = ans_np_flat[inds_inside]
    if good_ans_flat.size == 0:
        raise ValueError("There were no good anchors for an image of size {} with boxes {}".format(im_size, gt_boxes))

    # overlaps between the anchors and the gt boxes [num_anchors, num_gtboxes]
    overlaps = bbox_overlaps(good_ans_flat, gt_boxes)
    anchor_to_gtbox = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(anchor_to_gtbox.shape[0]), anchor_to_gtbox]
    gtbox_to_anchor = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gtbox_to_anchor, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # Good anchors are those that match SOMEWHERE within a decent tolerance
    # label: 1 is positive, 0 is negative, -1 is dont care.
    # assign bg labels first so that positive labels can clobber them
    labels = (-1) * np.ones(overlaps.shape[0], dtype=np.int64)
    labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

    # subsample positive labels if we have too many
    num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        labels[npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)] = -1

    # subsample negative labels if we have too many
    num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        labels[npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)] = -1
    # print("{} fg {} bg ratio{:.3f} inds inside {}".format(RPN_BATCHSIZE-num_bg, num_bg, (RPN_BATCHSIZE-num_bg)/RPN_BATCHSIZE, inds_inside.shape[0]))


    # Get the labels at the original size
    labels_unmap = (-1) * np.ones(ans_np_flat.shape[0], dtype=np.int64)
    labels_unmap[inds_inside] = labels

    # h, w, A
    labels_unmap_res = labels_unmap.reshape(ans_np.shape[:-1])
    anchor_inds = np.column_stack(np.where(labels_unmap_res >= 0))

    # These ought to be in the same order
    anchor_inds_flat = np.where(labels >= 0)[0]
    anchors = good_ans_flat[anchor_inds_flat]
    bbox_targets = gt_boxes[anchor_to_gtbox[anchor_inds_flat]]
    labels = labels[anchor_inds_flat]

    assert np.all(labels >= 0)


    # Anchors: [num_used, 4]
    # Anchor_inds: [num_used, 3] (h, w, A)
    # bbox_targets: [num_used, 4]
    # labels: [num_used]

    return anchors, anchor_inds, bbox_targets, labels
