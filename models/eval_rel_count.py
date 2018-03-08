"""
Baseline model that works by simply iterating through the training set to make a dictionary.

Also, caches this (we can use this for training).

The model is quite simple, so we don't use the base train/test code

"""
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
import numpy as np
import torch
import os
from lib.get_dataset_counts import get_counts, box_filter

from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, DATA_PATH, BOX_SCALE, IM_SCALE, PROPOSAL_FN
import torch.backends.cudnn as cudnn
from lib.pytorch_misc import optimistic_restore, nonintersecting_2d_inds
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from copy import deepcopy
import dill as pkl

cudnn.benchmark = True
conf = ModelConfig()

MUST_OVERLAP=False
train, val, test = VG.splits(num_val_im=conf.val_size, filter_non_overlap=MUST_OVERLAP,
                             filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals)
if conf.test:
    print("test data!")
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

fg_matrix, bg_matrix = get_counts(train_data=train, must_overlap=MUST_OVERLAP)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                          mode='rpntrain' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet,
                          nms_filter_duplicates=True, thresh=0.01)
detector.eval()
detector.cuda()

classifier = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                            mode='gtbox', use_resnet=conf.use_resnet,
                            nms_filter_duplicates=True, thresh=0.01)
classifier.eval()
classifier.cuda()

ckpt = torch.load(conf.ckpt)
mismatch = optimistic_restore(detector, ckpt['state_dict'])
mismatch = optimistic_restore(classifier, ckpt['state_dict'])

MOST_COMMON_MODE = True

if MOST_COMMON_MODE:
    prob_matrix = fg_matrix.astype(np.float32)
    prob_matrix[:,:,0] = bg_matrix

    # TRYING SOMETHING NEW.
    prob_matrix[:,:,0] += 1
    prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
    # prob_matrix /= float(fg_matrix.max())

    np.save(os.path.join(DATA_PATH, 'pred_stats.npy'), prob_matrix)
    prob_matrix[:,:,0] = 0 # Zero out BG
else:
    prob_matrix = fg_matrix.astype(np.float64)
    prob_matrix = prob_matrix / prob_matrix.max(2)[:,:,None]
    np.save(os.path.join(DATA_PATH, 'pred_dist.npy'), prob_matrix)

# It's test time!
def predict(boxes, classes):
    relation_possibilities_ = np.array(box_filter(boxes, must_overlap=MUST_OVERLAP), dtype=int)
    full_preds = np.zeros((boxes.shape[0], boxes.shape[0], train.num_predicates))
    for o1, o2 in relation_possibilities_:
        c1, c2 = classes[[o1, o2]]
        full_preds[o1, o2] = prob_matrix[c1, c2]

    full_preds[:,:,0] = 0.0 # Zero out BG.
    return full_preds

# ##########################################################################################
# ##########################################################################################

# For visualizing / exploring

c_to_ind = {c: i for i, c in enumerate(train.ind_to_classes)}
def gimme_the_dist(c1name, c2name):
    c1 = c_to_ind[c1name]
    c2 = c_to_ind[c2name]
    dist = prob_matrix[c1, c2]
    argz = np.argsort(-dist)
    for i, a in enumerate(argz):
        if dist[a] > 0.0:
            print("{:3d}: {:10s} ({:.4f})".format(i, train.ind_to_predicates[a], dist[a]))

counts = np.zeros((train.num_classes, train.num_classes, train.num_predicates), dtype=np.int64)
for ex_ind in tqdm(range(len(val))):
    gt_relations = val.relationships[ex_ind].copy()
    gt_classes = val.gt_classes[ex_ind].copy()
    o1o2 = gt_classes[gt_relations[:, :2]].tolist()
    for (o1, o2), pred in zip(o1o2, gt_relations[:, 2]):
        counts[o1, o2, pred] += 1

zeroshot_case = counts[np.where(prob_matrix == 0)].sum() / float(counts.sum())

max_inds = prob_matrix.argmax(2).ravel()
max_counts = counts.reshape(-1, 51)[np.arange(max_inds.shape[0]), max_inds]

most_freq_port = max_counts.sum()/float(counts.sum())


print(" Rel acc={:.2f}%, {:.2f}% zsl".format(
    most_freq_port*100, zeroshot_case*100))

# ##########################################################################################
# ##########################################################################################
T = len(val)
evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)

 # First do detection results
img_offset = 0
all_pred_entries = {'sgdet':[], 'sgcls':[], 'predcls':[]}
for val_b, b in enumerate(tqdm(val_loader)):

    det_result = detector[b]

    img_ids = b.gt_classes_primary.data.cpu().numpy()[:,0]
    scores_np = det_result.obj_scores.data.cpu().numpy()
    cls_preds_np = det_result.obj_preds.data.cpu().numpy()
    boxes_np = det_result.boxes_assigned.data.cpu().numpy()* BOX_SCALE/IM_SCALE
    # boxpriors_np = det_result.box_priors.data.cpu().numpy()
    im_inds_np = det_result.im_inds.data.cpu().numpy() + img_offset

    for img_i in np.unique(img_ids + img_offset):
        gt_entry = {
            'gt_classes': val.gt_classes[img_i].copy(),
            'gt_relations': val.relationships[img_i].copy(),
            'gt_boxes': val.gt_boxes[img_i].copy(),
        }

        pred_boxes = boxes_np[im_inds_np == img_i]
        pred_classes = cls_preds_np[im_inds_np == img_i]
        obj_scores = scores_np[im_inds_np == img_i]

        all_rels = nonintersecting_2d_inds(pred_boxes.shape[0])
        fp = predict(pred_boxes, pred_classes)
        fp_pred = fp[all_rels[:,0], all_rels[:,1]]

        scores = np.column_stack((
            obj_scores[all_rels[:,0]],
            obj_scores[all_rels[:,1]],
            fp_pred.max(1)
        )).prod(1)
        sorted_inds = np.argsort(-scores)
        sorted_inds = sorted_inds[scores[sorted_inds] > 0] #[:100]
        pred_entry = {
            'pred_boxes': pred_boxes,
            'pred_classes': pred_classes,
            'obj_scores': obj_scores,
            'pred_rel_inds': all_rels[sorted_inds],
            'rel_scores': fp_pred[sorted_inds],
        }
        all_pred_entries['sgdet'].append(pred_entry)
        evaluator['sgdet'].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    img_offset += img_ids.max() + 1
evaluator['sgdet'].print_stats()

# -----------------------------------------------------------------------------------------
# EVAL CLS AND SG

img_offset = 0
for val_b, b in enumerate(tqdm(val_loader)):

    det_result = classifier[b]
    scores, cls_preds = det_result.rm_obj_dists[:,1:].data.max(1)
    scores_np = scores.cpu().numpy()
    cls_preds_np = (cls_preds+1).cpu().numpy()

    img_ids = b.gt_classes_primary.data.cpu().numpy()[:,0]
    boxes_np = b.gt_boxes_primary.data.cpu().numpy()
    im_inds_np = det_result.im_inds.data.cpu().numpy() + img_offset

    for img_i in np.unique(img_ids + img_offset):
        gt_entry = {
            'gt_classes': val.gt_classes[img_i].copy(),
            'gt_relations': val.relationships[img_i].copy(),
            'gt_boxes': val.gt_boxes[img_i].copy(),
        }

        pred_boxes = boxes_np[im_inds_np == img_i]
        pred_classes = cls_preds_np[im_inds_np == img_i]
        obj_scores = scores_np[im_inds_np == img_i]

        all_rels = nonintersecting_2d_inds(pred_boxes.shape[0])
        fp = predict(pred_boxes, pred_classes)
        fp_pred = fp[all_rels[:,0], all_rels[:,1]]

        sg_cls_scores = np.column_stack((
            obj_scores[all_rels[:,0]],
            obj_scores[all_rels[:,1]],
            fp_pred.max(1)
        )).prod(1)
        sg_cls_inds = np.argsort(-sg_cls_scores)
        sg_cls_inds = sg_cls_inds[sg_cls_scores[sg_cls_inds] > 0] #[:100]

        pred_entry = {
            'pred_boxes': pred_boxes,
            'pred_classes': pred_classes,
            'obj_scores': obj_scores,
            'pred_rel_inds': all_rels[sg_cls_inds],
            'rel_scores': fp_pred[sg_cls_inds],
        }
        all_pred_entries['sgcls'].append(deepcopy(pred_entry))
        evaluator['sgcls'].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

        ########################################################
        fp = predict(gt_entry['gt_boxes'], gt_entry['gt_classes'])
        fp_pred = fp[all_rels[:, 0], all_rels[:, 1]]

        pred_cls_scores = fp_pred.max(1)
        pred_cls_inds = np.argsort(-pred_cls_scores)
        pred_cls_inds = pred_cls_inds[pred_cls_scores[pred_cls_inds] > 0][:100]

        pred_entry['pred_rel_inds'] = all_rels[pred_cls_inds]
        pred_entry['rel_scores'] = fp_pred[pred_cls_inds]
        pred_entry['pred_classes'] = gt_entry['gt_classes']
        pred_entry['obj_scores'] = np.ones(pred_entry['pred_classes'].shape[0])

        all_pred_entries['predcls'].append(pred_entry)

        evaluator['predcls'].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    img_offset += img_ids.max() + 1
evaluator['predcls'].print_stats()
evaluator['sgcls'].print_stats()

for mode, entries in all_pred_entries.items():
    with open('caches/freqbaseline-{}-{}.pkl'.format('overlap' if MUST_OVERLAP else 'nonoverlap', mode), 'wb') as f:
        pkl.dump(entries, f)
