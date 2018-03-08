# JUST TO CHECK THAT IT IS EXACTLY THE SAME..................................
import numpy as np
from config import MODES

class BasicSceneGraphEvaluator:

    def __init__(self, mode):
        self.result_dict = {}
        self.mode = {'sgdet':'sg_det', 'sgcls':'sg_cls', 'predcls':'pred_cls'}[mode]

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {20:[], 50:[], 100:[]}


    @classmethod
    def all_modes(cls):
        evaluators = {m: cls(mode=m) for m in MODES}
        return evaluators
    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, iou_thresh=0.5):

        roidb_entry = {
            'max_overlaps': np.ones(gt_entry['gt_classes'].shape[0], dtype=np.int64),
            'boxes': gt_entry['gt_boxes'],
            'gt_relations': gt_entry['gt_relations'],
            'gt_classes': gt_entry['gt_classes'],
        }
        sg_entry = {
            'boxes': pred_entry['pred_boxes'],
            'relations': pred_entry['pred_rels'],
            'obj_scores': pred_entry['obj_scores'],
            'rel_scores': pred_entry['rel_scores'],
            'pred_classes': pred_entry['pred_classes'],
        }

        pred_triplets, triplet_boxes = \
            eval_relation_recall(sg_entry, roidb_entry,
                                self.result_dict,
                                self.mode,
                                iou_thresh=iou_thresh)
        return pred_triplets, triplet_boxes


    def save(self, fn):
        np.save(fn, self.result_dict)


    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))


def eval_relation_recall(sg_entry,
                         roidb_entry,
                         result_dict,
                         mode,
                         iou_thresh):

    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_relations[:,2],
                                             gt_relations[:,:2],
                                             gt_classes,
                                             gt_boxes,
                                             gt_predicate_scores,
                                             gt_class_scores)

    # pred
    box_preds = sg_entry['boxes']
    num_boxes = box_preds.shape[0]
    relations = sg_entry['relations']
    classes = sg_entry['pred_classes'].copy()
    class_scores = sg_entry['obj_scores'].copy()

    num_relations = relations.shape[0]

    if mode =='pred_cls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert(num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode =='sg_cls':
        assert(num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        # classes = np.argmax(class_preds, 1)
        # class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode =='sg_det':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        # classes = np.argmax(class_preds, 1)
        # class_scores = class_preds.max(axis=1)
        boxes = box_preds
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets = np.column_stack((
        classes[relations[:, 0]],
        relations[:,2],
        classes[relations[:, 1]],
    ))
    pred_triplet_boxes = np.column_stack((
        boxes[relations[:, 0]],
        boxes[relations[:, 1]],
    ))
    relation_scores = np.column_stack((
        class_scores[relations[:, 0]],
        sg_entry['rel_scores'],
        class_scores[relations[:, 1]],
    )).prod(1)

    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds,:],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds,:],
                                  iou_thresh)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):

    # format predictions into triplets
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in range(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i,:2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score =  class_scores[sub_i]
        score *= class_scores[obj_i]
        score *= predicate_scores[i]
        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh):

    # compute the R@K metric for a set of predicted triplets

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep,:]
        sub_iou = iou(gt_box[:4], boxes[:,:4])
        obj_iou = iou(gt_box[4:], boxes[:,4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt) / float(num_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps
