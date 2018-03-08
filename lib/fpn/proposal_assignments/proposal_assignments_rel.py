# --------------------------------------------------------
# Goal: assign ROIs to targets
# --------------------------------------------------------


import numpy as np
import numpy.random as npr
from config import BG_THRESH_HI, BG_THRESH_LO, FG_FRACTION_REL, ROIS_PER_IMG_REL, REL_FG_FRACTION, \
    RELS_PER_IMG
from lib.fpn.box_utils import bbox_overlaps
from lib.pytorch_misc import to_variable, nonintersecting_2d_inds
from collections import defaultdict
import torch


@to_variable
def proposal_assignments_rel(rpn_rois, gt_boxes, gt_classes, gt_rels, image_offset, fg_thresh=0.5):
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
    fg_rois_per_image = int(np.round(ROIS_PER_IMG_REL * FG_FRACTION_REL))
    fg_rels_per_image = int(np.round(REL_FG_FRACTION * RELS_PER_IMG))

    pred_inds_np = rpn_rois[:, 0].cpu().numpy().astype(np.int64)
    pred_boxes_np = rpn_rois[:, 1:].cpu().numpy()
    gt_boxes_np = gt_boxes.cpu().numpy()
    gt_classes_np = gt_classes.cpu().numpy()
    gt_rels_np = gt_rels.cpu().numpy()

    gt_classes_np[:, 0] -= image_offset
    gt_rels_np[:, 0] -= image_offset

    num_im = gt_classes_np[:, 0].max()+1

    rois = []
    obj_labels = []
    rel_labels = []
    bbox_targets = []

    num_box_seen = 0

    for im_ind in range(num_im):
        pred_ind = np.where(pred_inds_np == im_ind)[0]

        gt_ind = np.where(gt_classes_np[:, 0] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[gt_rels_np[:, 0] == im_ind, 1:]

        pred_boxes_i = np.concatenate((pred_boxes_np[pred_ind], gt_boxes_i), 0)
        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
 
        obj_inds_i, obj_labels_i, obj_assignments_i = _sel_inds(ious, gt_classes_i, 
            fg_thresh, fg_rois_per_image, ROIS_PER_IMG_REL)

        all_rels_i = _sel_rels(ious[obj_inds_i], pred_boxes_i[obj_inds_i], obj_labels_i,
                               gt_classes_i, gt_rels_i,
                               fg_thresh=fg_thresh, fg_rels_per_image=fg_rels_per_image)
        all_rels_i[:,0:2] += num_box_seen

        rois.append(np.column_stack((
            im_ind * np.ones(obj_inds_i.shape[0], dtype=np.float32),
            pred_boxes_i[obj_inds_i],
        )))
        obj_labels.append(obj_labels_i)
        rel_labels.append(np.column_stack((
            im_ind*np.ones(all_rels_i.shape[0], dtype=np.int64),
            all_rels_i,
        )))

        # print("Gtboxes i {} obj assignments i {}".format(gt_boxes_i, obj_assignments_i))
        bbox_targets.append(gt_boxes_i[obj_assignments_i])

        num_box_seen += obj_inds_i.size

    rois = torch.FloatTensor(np.concatenate(rois, 0)).cuda(rpn_rois.get_device(), async=True)
    labels = torch.LongTensor(np.concatenate(obj_labels, 0)).cuda(rpn_rois.get_device(), async=True)
    bbox_targets = torch.FloatTensor(np.concatenate(bbox_targets, 0)).cuda(rpn_rois.get_device(),
                                                                           async=True)
    rel_labels = torch.LongTensor(np.concatenate(rel_labels, 0)).cuda(rpn_rois.get_device(),
                                                                      async=True)

    return rois, labels, bbox_targets, rel_labels


def _sel_rels(ious, pred_boxes, pred_labels, gt_classes, gt_rels, fg_thresh=0.5, fg_rels_per_image=128, num_sample_per_gt=1, filter_non_overlap=True):
    """
    Selects the relations needed
    :param ious: [num_pred', num_gt]
    :param pred_boxes: [num_pred', num_gt]
    :param pred_labels: [num_pred']
    :param gt_classes: [num_gt]
    :param gt_rels: [num_gtrel, 3]
    :param fg_thresh: 
    :param fg_rels_per_image: 
    :return: new rels, [num_predrel, 3] where each is (pred_ind1, pred_ind2, predicate)
    """
    is_match = (ious >= fg_thresh) & (pred_labels[:, None] == gt_classes[None, :])

    pbi_iou = bbox_overlaps(pred_boxes, pred_boxes)

    # Limit ourselves to only IOUs that overlap, but are not the exact same box
    # since we duplicated stuff earlier.
    if filter_non_overlap:
        rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
        rels_intersect = rel_possibilities
    else:
        rel_possibilities = np.ones((pred_labels.shape[0], pred_labels.shape[0]),
                                    dtype=np.int64) - np.eye(pred_labels.shape[0], dtype=np.int64)
        rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)

    # ONLY select relations between ground truth because otherwise we get useless data
    rel_possibilities[pred_labels == 0] = 0
    rel_possibilities[:,pred_labels == 0] = 0

    # For each GT relationship, sample exactly 1 relationship.
    fg_rels = []
    p_size = []
    for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels):
        fg_rels_i = []
        fg_scores_i = []

        for from_ind in np.where(is_match[:,from_gtind])[0]:
            for to_ind in np.where(is_match[:,to_gtind])[0]:
                if from_ind != to_ind:
                    fg_rels_i.append((from_ind, to_ind, rel_id))
                    fg_scores_i.append((ious[from_ind, from_gtind]*ious[to_ind, to_gtind]))
                    rel_possibilities[from_ind, to_ind] = 0
        if len(fg_rels_i) == 0:
            continue
        p = np.array(fg_scores_i)
        p = p/p.sum()
        p_size.append(p.shape[0])
        num_to_add = min(p.shape[0], num_sample_per_gt)
        for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
            fg_rels.append(fg_rels_i[rel_to_add])

    bg_rels = np.column_stack(np.where(rel_possibilities))
    bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))

    fg_rels = np.array(fg_rels, dtype=np.int64)
    if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
        fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        # print("{} scores for {} GT. max={} min={} BG rels {}".format(
        #     fg_rels_scores.shape[0], gt_rels.shape[0], fg_rels_scores.max(), fg_rels_scores.min(),
        #     bg_rels.shape))
    elif fg_rels.size == 0:
        fg_rels = np.zeros((0,3), dtype=np.int64)

    num_bg_rel = min(RELS_PER_IMG - fg_rels.shape[0], bg_rels.shape[0])
    if bg_rels.size > 0:

        # Sample 4x as many intersecting relationships as non-intersecting.
        bg_rels_intersect = rels_intersect[bg_rels[:,0], bg_rels[:,1]]
        p = bg_rels_intersect.astype(np.float32)
        p[bg_rels_intersect == 0] = 0.2
        p[bg_rels_intersect == 1] = 0.8
        p /= p.sum()
        bg_rels = bg_rels[np.random.choice(bg_rels.shape[0], p=p, size=num_bg_rel, replace=False)]
    else:
        bg_rels = np.zeros((0,3), dtype=np.int64)

    #print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))

    all_rels = np.concatenate((fg_rels, bg_rels), 0)

    # Sort by 2nd ind and then 1st ind
    all_rels = all_rels[np.lexsort((all_rels[:, 1], all_rels[:, 0]))]
    return all_rels

def _sel_inds(ious, gt_classes_i, fg_thresh=0.5, fg_rois_per_image=128, rois_per_image=256, n_sample_per=1):

    #gt_assignment = ious.argmax(1)
    #max_overlaps = ious[np.arange(ious.shape[0]), gt_assignment]
    #fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    
    fg_ious = ious.T >= fg_thresh #[num_gt, num_pred]
    #is_bg = ~fg_ious.any(0)

    # Sample K inds per GT image.
    fg_inds = []
    for i, (ious_i, cls_i) in enumerate(zip(fg_ious, gt_classes_i)):
        n_sample_this_roi = min(n_sample_per, ious_i.sum())
        if n_sample_this_roi > 0:
            p = ious_i.astype(np.float64) / ious_i.sum()
            for ind in npr.choice(ious_i.shape[0], p=p, size=n_sample_this_roi, replace=False):
                fg_inds.append((ind, i))
    
    fg_inds = np.array(fg_inds, dtype=np.int64)
    if fg_inds.size == 0:
        fg_inds = np.zeros((0, 2), dtype=np.int64)
    elif fg_inds.shape[0] > fg_rois_per_image:
        #print("sample FG")
        fg_inds = fg_inds[npr.choice(fg_inds.shape[0], size=fg_rois_per_image, replace=False)]
    
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    max_overlaps = ious.max(1)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >= BG_THRESH_LO))[0]

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = min(rois_per_image-fg_inds.shape[0], bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # FIx for format issues
    obj_inds = np.concatenate((fg_inds[:,0], bg_inds), 0)
    obj_assignments_i = np.concatenate((fg_inds[:,1], np.zeros(bg_inds.shape[0], dtype=np.int64)))
    obj_labels_i = gt_classes_i[obj_assignments_i]
    obj_labels_i[fg_inds.shape[0]:] = 0
    #print("{} FG and {} BG".format(fg_inds.shape[0], bg_inds.shape[0]))
    return obj_inds, obj_labels_i, obj_assignments_i


