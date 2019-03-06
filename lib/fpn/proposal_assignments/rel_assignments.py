# --------------------------------------------------------
# Goal: assign ROIs to targets
# --------------------------------------------------------


import numpy as np
import numpy.random as npr
from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose
from lib.fpn.box_utils import bbox_preds, center_size, bbox_overlaps
import torch
from lib.pytorch_misc import diagonal_inds, to_variable
from config import RELS_PER_IMG, REL_FG_FRACTION
@to_variable
def rel_assignments_sgdet(im_inds, rpn_rois, roi_gtlabels, gt_boxes, gt_classes, gt_rels, image_offset,
                    fg_thresh=0.5, num_sample_per_gt=4, filter_non_overlap=True):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    fg_rels_per_image = int(np.round(REL_FG_FRACTION * 64))

    pred_inds_np = im_inds.cpu().numpy()
    pred_boxes_np = rpn_rois.cpu().numpy()
    pred_boxlabels_np = roi_gtlabels.cpu().numpy()
    gt_boxes_np = gt_boxes.cpu().numpy()
    gt_classes_np = gt_classes.cpu().numpy()
    gt_rels_np = gt_rels.cpu().numpy()

    gt_classes_np[:, 0] -= image_offset
    gt_rels_np[:, 0] -= image_offset

    num_im = gt_classes_np[:, 0].max()+1

    # print("Pred inds {} pred boxes {} pred box labels {} gt classes {} gt rels {}".format(
    #     pred_inds_np, pred_boxes_np, pred_boxlabels_np, gt_classes_np, gt_rels_np
    # ))

    rel_labels = []
    num_box_seen = 0
    for im_ind in range(num_im):
        pred_ind = np.where(pred_inds_np == im_ind)[0]

        gt_ind = np.where(gt_classes_np[:, 0] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[gt_rels_np[:, 0] == im_ind, 1:]

        # [num_pred, num_gt]
        pred_boxes_i = pred_boxes_np[pred_ind]
        pred_boxlabels_i = pred_boxlabels_np[pred_ind]

        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
        is_match = (pred_boxlabels_i[:,None] == gt_classes_i[None]) & (ious >= fg_thresh)

        # FOR BG. Limit ourselves to only IOUs that overlap, but are not the exact same box
        pbi_iou = bbox_overlaps(pred_boxes_i, pred_boxes_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_boxes_i.shape[0], pred_boxes_i.shape[0]),
                                        dtype=np.int64) - np.eye(pred_boxes_i.shape[0],
                                                                 dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)

        # ONLY select relations between ground truth because otherwise we get useless data
        rel_possibilities[pred_boxlabels_i == 0] = 0
        rel_possibilities[:, pred_boxlabels_i == 0] = 0

        # Sample the GT relationships.
        fg_rels = []
        p_size = []
        for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels_i):
            fg_rels_i = []
            fg_scores_i = []

            for from_ind in np.where(is_match[:, from_gtind])[0]:
                for to_ind in np.where(is_match[:, to_gtind])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append((from_ind, to_ind, rel_id))
                        fg_scores_i.append((ious[from_ind, from_gtind] * ious[to_ind, to_gtind]))
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue
            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])

        fg_rels = np.array(fg_rels, dtype=np.int64)
        if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
            fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        elif fg_rels.size == 0:
            fg_rels = np.zeros((0, 3), dtype=np.int64)

        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))

        num_bg_rel = min(64 - fg_rels.shape[0], bg_rels.shape[0])
        if bg_rels.size > 0:
            # Sample 4x as many intersecting relationships as non-intersecting.
            # bg_rels_intersect = rels_intersect[bg_rels[:, 0], bg_rels[:, 1]]
            # p = bg_rels_intersect.astype(np.float32)
            # p[bg_rels_intersect == 0] = 0.2
            # p[bg_rels_intersect == 1] = 0.8
            # p /= p.sum()
            bg_rels = bg_rels[
                np.random.choice(bg_rels.shape[0],
                                 #p=p,
                                 size=num_bg_rel, replace=False)]
        else:
            bg_rels = np.zeros((0, 3), dtype=np.int64)

        if fg_rels.size == 0 and bg_rels.size == 0:
            # Just put something here
            bg_rels = np.array([[0, 0, 0]], dtype=np.int64)

        # print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:,0:2] += num_box_seen

        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:,1], all_rels_i[:,0]))]

        rel_labels.append(np.column_stack((
            im_ind*np.ones(all_rels_i.shape[0], dtype=np.int64),
            all_rels_i,
        )))

        num_box_seen += pred_boxes_i.shape[0]
    rel_labels = torch.LongTensor(np.concatenate(rel_labels, 0)).cuda(rpn_rois.get_device(),
                                                                      async=True)
    return rel_labels

@to_variable
def rel_assignments_sgcls(rois, gt_boxes, gt_classes, gt_rels, image_offset):
    """
    sample_rels to balance proportion of positive and negative samples
    :param rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]. Not needed it seems
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        Note, the img_inds here start at image_offset
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type].
        Note, the img_inds here start at image_offset
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    im_inds = rois[:,0].long()

    num_im = im_inds[-1] + 1

    # Offset the image indices in fg_rels to refer to absolute indices (not just within img i)
    fg_rels = gt_rels.clone()
    fg_rels[:,0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]

    # Try ALL things, not just intersections.
    is_cand = (im_inds[:, None] == im_inds[None])
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0

    # # Compute salience
    # gt_inds = fg_rels[:, 1:3].contiguous().view(-1)
    # labels_arange = labels.data.new(labels.size(0))
    # torch.arange(0, labels.size(0), out=labels_arange)
    # salience_labels = ((gt_inds[:, None] == labels_arange[None]).long().sum(0) > 0).long()
    # labels = torch.stack((labels, salience_labels), 1)

    # Add in some BG labels

    # NOW WE HAVE TO EXCLUDE THE FGs.
    # TODO: check if this causes an error if many duplicate GTs havent been filtered out

    is_cand.view(-1)[fg_rels[:,1]*im_inds.size(0) + fg_rels[:,2]] = 0
    is_bgcand = is_cand.nonzero()
    # TODO: make this sample on a per image case
    # If too many then sample
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im))
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)

    # If too many then sample
    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0,
                 int(RELS_PER_IMG * num_im) - num_fg)
    if num_bg > 0:
        bg_rels = torch.cat((
            im_inds[is_bgcand[:, 0]][:, None],
            is_bgcand,
            (is_bgcand[:, 0, None] < -10).long(),
        ), 1)

        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)
        rel_labels = torch.cat((fg_rels, bg_rels), 0)
    else:
        rel_labels = fg_rels


    # last sort by rel.
    _, perm = torch.sort(rel_labels[:, 0]*(gt_boxes.size(0)**2) +
                         rel_labels[:,1]*gt_boxes.size(0) + rel_labels[:,2])

    rel_labels = rel_labels[perm].contiguous()

    labels = gt_classes[:,1].contiguous()
    return rois, labels, rel_labels
