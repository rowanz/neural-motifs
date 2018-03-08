"""
Training script 4 Detection
"""
from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

cudnn.benchmark = True
conf = ModelConfig()

if conf.coco:
    train, val = CocoDetection.splits()
    val.ids = val.ids[:conf.val_size]
    train.ids = train.ids
    train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     num_gpus=conf.num_gpus)
else:
    train, val, _ = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                              filter_empty_rels=False, use_proposals=conf.use_proposals)
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                          mode='rpntrain' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet)
detector.cuda()

# Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers
if conf.use_proposals:
    for n, param in detector.named_parameters():
        if n.startswith('features'):
            param.requires_grad = False

optimizer = optim.SGD([p for p in detector.parameters() if p.requires_grad],
                      weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                              verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)

start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)
    if optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = ckpt['epoch']


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)

    :return:
    """
    result = detector[b]
    scores = result.od_obj_dists
    box_deltas = result.od_box_deltas
    labels = result.od_obj_labels
    roi_boxes = result.od_box_priors
    bbox_targets = result.od_box_targets
    rpn_scores = result.rpn_scores
    rpn_box_deltas = result.rpn_box_deltas

    # detector loss
    valid_inds = (labels.data != 0).nonzero().squeeze(1)
    fg_cnt = valid_inds.size(0)
    bg_cnt = labels.size(0) - fg_cnt
    class_loss = F.cross_entropy(scores, labels)

    # No gather_nd in pytorch so instead convert first 2 dims of tensor to 1d
    box_reg_mult = 2 * (1. / FG_FRACTION) * fg_cnt / (fg_cnt + bg_cnt + 1e-4)
    twod_inds = valid_inds * box_deltas.size(1) + labels[valid_inds].data

    box_loss = bbox_loss(roi_boxes[valid_inds], box_deltas.view(-1, 4)[twod_inds],
                         bbox_targets[valid_inds]) * box_reg_mult

    loss = class_loss + box_loss

    # RPN loss
    if not conf.use_proposals:
        train_anchor_labels = b.train_anchor_labels[:, -1]
        train_anchors = b.train_anchors[:, :4]
        train_anchor_targets = b.train_anchors[:, 4:]

        train_valid_inds = (train_anchor_labels.data == 1).nonzero().squeeze(1)
        rpn_class_loss = F.cross_entropy(rpn_scores, train_anchor_labels)

        # print("{} fg {} bg, ratio of {:.3f} vs {:.3f}. RPN {}fg {}bg ratio of {:.3f} vs {:.3f}".format(
        #     fg_cnt, bg_cnt, fg_cnt / (fg_cnt + bg_cnt + 1e-4), FG_FRACTION,
        #     train_valid_inds.size(0), train_anchor_labels.size(0)-train_valid_inds.size(0),
        #     train_valid_inds.size(0) / (train_anchor_labels.size(0) + 1e-4), RPN_FG_FRACTION), flush=True)
        rpn_box_mult = 2 * (1. / RPN_FG_FRACTION) * train_valid_inds.size(0) / (train_anchor_labels.size(0) + 1e-4)
        rpn_box_loss = bbox_loss(train_anchors[train_valid_inds],
                                 rpn_box_deltas[train_valid_inds],
                                 train_anchor_targets[train_valid_inds]) * rpn_box_mult

        loss += rpn_class_loss + rpn_box_loss
        res = pd.Series([rpn_class_loss.data[0], rpn_box_loss.data[0],
                         class_loss.data[0], box_loss.data[0], loss.data[0]],
                        ['rpn_class_loss', 'rpn_box_loss', 'class_loss', 'box_loss', 'total'])
    else:
        res = pd.Series([class_loss.data[0], box_loss.data[0], loss.data[0]],
                        ['class_loss', 'box_loss', 'total'])

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, clip=True)
    optimizer.step()

    return res


def val_epoch():
    detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    for val_b, batch in enumerate(val_loader):
        vr.append(val_batch(val_b, batch))
    vr = np.concatenate(vr, 0)
    if vr.shape[0] == 0:
        print("No detections anywhere")
        return 0.0

    val_coco = val.coco
    coco_dt = val_coco.loadRes(vr)
    coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = val.ids if conf.coco else [x for x in range(len(val))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]
    return mAp


def val_batch(batch_num, b):
    result = detector[b]
    if result is None:
        return np.zeros((0, 7))
    scores_np = result.obj_scores.data.cpu().numpy()
    cls_preds_np = result.obj_preds.data.cpu().numpy()
    boxes_np = result.boxes_assigned.data.cpu().numpy()
    im_inds_np = result.im_inds.data.cpu().numpy()
    im_scales = b.im_sizes.reshape((-1, 3))[:, 2]
    if conf.coco:
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        cls_preds_np[:] = [val.ind_to_id[c_ind] for c_ind in cls_preds_np]
        im_inds_np[:] = [val.ids[im_ind + batch_num * conf.batch_size * conf.num_gpus]
                         for im_ind in im_inds_np]
    else:
        boxes_np *= BOX_SCALE / IM_SCALE
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        im_inds_np += batch_num * conf.batch_size * conf.num_gpus

    return np.column_stack((im_inds_np, boxes_np, scores_np, cls_preds_np))


print("Training starts now!")
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    mAp = val_epoch()
    scheduler.step(mAp)

    torch.save({
        'epoch': epoch,
        'state_dict': detector.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(conf.save_dir, '{}-{}.tar'.format('coco' if conf.coco else 'vg', epoch)))
