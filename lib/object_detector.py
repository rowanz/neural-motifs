import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F

from config import ANCHOR_SIZE, ANCHOR_RATIOS, ANCHOR_SCALES
from lib.fpn.generate_anchors import generate_anchors
from lib.fpn.box_utils import bbox_preds, center_size, bbox_overlaps
from lib.fpn.nms.functions.nms import apply_nms
from lib.fpn.proposal_assignments.proposal_assignments_gtbox import proposal_assignments_gtbox
from lib.fpn.proposal_assignments.proposal_assignments_det import proposal_assignments_det

from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.pytorch_misc import enumerate_by_image, gather_nd, diagonal_inds, Flattener
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet101
from torch.nn.parallel._functions import Gather


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""

    def __init__(self, od_obj_dists=None, rm_obj_dists=None,
                 obj_scores=None, obj_preds=None, obj_fmap=None,
                 od_box_deltas=None, rm_box_deltas=None,
                 od_box_targets=None, rm_box_targets=None, od_box_priors=None, rm_box_priors=None,
                 boxes_assigned=None, boxes_all=None, od_obj_labels=None, rm_obj_labels=None,
                 rpn_scores=None, rpn_box_deltas=None, rel_labels=None,
                 im_inds=None, fmap=None, rel_dists=None, rel_inds=None, rel_rep=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([v is None for k, v in self.__dict__.items() if k != 'self'])


def gather_res(outputs, target_device, dim=0):
    """
    Assuming the signatures are the same accross results!
    """
    out = outputs[0]
    args = {field: Gather.apply(target_device, dim, *[getattr(o, field) for o in outputs])
            for field, v in out.__dict__.items() if v is not None}
    return type(out)(**args)


class ObjectDetector(nn.Module):
    """
    Core model for doing object detection + getting the visual features. This could be the first step in
    a pipeline. We can provide GT rois or use the RPN (which would then be classification!)
    """
    MODES = ('rpntrain', 'gtbox', 'refinerels', 'proposals')

    def __init__(self, classes, mode='rpntrain', num_gpus=1, nms_filter_duplicates=True,
                 max_per_img=64, use_resnet=False, thresh=0.05):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(ObjectDetector, self).__init__()

        if mode not in self.MODES:
            raise ValueError("invalid mode")
        self.mode = mode

        self.classes = classes
        self.num_gpus = num_gpus
        self.pooling_size = 7
        self.nms_filter_duplicates = nms_filter_duplicates
        self.max_per_img = max_per_img
        self.use_resnet = use_resnet
        self.thresh = thresh

        if not self.use_resnet:
            vgg_model = load_vgg()
            self.features = vgg_model.features
            self.roi_fmap = vgg_model.classifier
            rpn_input_dim = 512
            output_dim = 4096
        else:  # Deprecated
            self.features = load_resnet()
            self.compress = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
            )
            self.roi_fmap = nn.Sequential(
                nn.Linear(256 * 7 * 7, 2048),
                nn.SELU(inplace=True),
                nn.AlphaDropout(p=0.05),
                nn.Linear(2048, 2048),
                nn.SELU(inplace=True),
                nn.AlphaDropout(p=0.05),
            )
            rpn_input_dim = 1024
            output_dim = 2048

        self.score_fc = nn.Linear(output_dim, self.num_classes)
        self.bbox_fc = nn.Linear(output_dim, self.num_classes * 4)
        self.rpn_head = RPNHead(dim=512, input_dim=rpn_input_dim)

    @property
    def num_classes(self):
        return len(self.classes)

    def feature_map(self, x):
        """
        Produces feature map from the input image
        :param x: [batch_size, 3, size, size] float32 padded image
        :return: Feature maps at 1/16 the original size.
        Each one is [batch_size, dim, IM_SIZE/k, IM_SIZE/k].
        """
        if not self.use_resnet:
            return self.features(x)  # Uncomment this for "stanford" setting in which it's frozen:      .detach()
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        c2 = self.features.layer1(x)
        c3 = self.features.layer2(c2)
        c4 = self.features.layer3(c3)
        return c4

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            self.compress(features) if self.use_resnet else features, rois)
        return self.roi_fmap(feature_pool.view(rois.size(0), -1))

    def rpn_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
                  train_anchor_inds=None, proposals=None):
        """
        Gets boxes from the RPN
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        rpn_feats = self.rpn_head(fmap)
        rois = self.rpn_head.roi_proposals(
            rpn_feats, im_sizes, nms_thresh=0.7,
            pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000,
            post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000,
        )
        if self.training:
            if gt_boxes is None or gt_classes is None or train_anchor_inds is None:
                raise ValueError(
                    "Must supply GT boxes, GT classes, trainanchors when in train mode")
            rpn_scores, rpn_box_deltas = self.rpn_head.anchor_preds(rpn_feats, train_anchor_inds,
                                                                    image_offset)

            if gt_rels is not None and self.mode == 'rpntrain':
                raise ValueError("Training the object detector and the relationship model with detection"
                                 "at the same time isn't supported")

            if self.mode == 'refinerels':
                all_rois = Variable(rois)
                # Potentially you could add in GT rois if none match
                # is_match = (bbox_overlaps(rois[:,1:].contiguous(), gt_boxes.data) > 0.5).long()
                # gt_not_matched = (is_match.sum(0) == 0).nonzero()
                #
                # if gt_not_matched.dim() > 0:
                #     gt_to_add = torch.cat((gt_classes[:,0,None][gt_not_matched.squeeze(1)].float(),
                #                            gt_boxes[gt_not_matched.squeeze(1)]), 1)
                #
                #     all_rois = torch.cat((all_rois, gt_to_add),0)
                #     num_gt = gt_to_add.size(0)
                labels = None
                bbox_targets = None
                rel_labels = None
            else:
                all_rois, labels, bbox_targets = proposal_assignments_det(
                    rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)
                rel_labels = None

        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None
            rel_labels = None
            rpn_box_deltas = None
            rpn_scores = None

        return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels

    def gt_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
                 train_anchor_inds=None, proposals=None):
        """
        Gets GT boxes!
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        assert gt_boxes is not None
        im_inds = gt_classes[:, 0] - image_offset
        rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, labels, rel_labels = proposal_assignments_gtbox(
                rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset,
                fg_thresh=0.5)
        else:
            labels = gt_classes[:, 1]
            rel_labels = None

        return rois, labels, None, None, None, rel_labels

    def proposal_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
                       train_anchor_inds=None, proposals=None):
        """
        Gets boxes from the RPN
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        assert proposals is not None

        rois = filter_roi_proposals(proposals[:, 2:].data.contiguous(), proposals[:, 1].data.contiguous(),
                                    np.array([2000] * len(im_sizes)),
                                    nms_thresh=0.7,
                                    pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000,
                                    post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000,
                                    )
        if self.training:
            all_rois, labels, bbox_targets = proposal_assignments_det(
                rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)

            # RETRAINING FOR DETECTION HERE.
            all_rois = torch.cat((all_rois, Variable(rois)), 0)
        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None

        rpn_scores = None
        rpn_box_deltas = None
        rel_labels = None

        return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels

    def get_boxes(self, *args, **kwargs):
        if self.mode == 'gtbox':
            fn = self.gt_boxes
        elif self.mode == 'proposals':
            assert kwargs['proposals'] is not None
            fn = self.proposal_boxes
        else:
            fn = self.rpn_boxes
        return fn(*args, **kwargs)

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param proposals: things
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
        """
        fmap = self.feature_map(x)

        # Get boxes from RPN
        rois, obj_labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels = \
            self.get_boxes(fmap, im_sizes, image_offset, gt_boxes,
                           gt_classes, gt_rels, train_anchor_inds, proposals=proposals)

        # Now classify them
        obj_fmap = self.obj_feature_map(fmap, rois)
        od_obj_dists = self.score_fc(obj_fmap)
        od_box_deltas = self.bbox_fc(obj_fmap).view(
            -1, len(self.classes), 4) if self.mode != 'gtbox' else None

        od_box_priors = rois[:, 1:]

        if (not self.training and not self.mode == 'gtbox') or self.mode in ('proposals', 'refinerels'):
            nms_inds, nms_scores, nms_preds, nms_boxes_assign, nms_boxes, nms_imgs = self.nms_boxes(
                od_obj_dists,
                rois,
                od_box_deltas, im_sizes,
            )
            im_inds = nms_imgs + image_offset
            obj_dists = od_obj_dists[nms_inds]
            obj_fmap = obj_fmap[nms_inds]
            box_deltas = od_box_deltas[nms_inds]
            box_priors = nms_boxes[:, 0]

            if self.training and not self.mode == 'gtbox':
                # NOTE: If we're doing this during training, we need to assign labels here.
                pred_to_gtbox = bbox_overlaps(box_priors, gt_boxes).data
                pred_to_gtbox[im_inds.data[:, None] != gt_classes.data[None, :, 0]] = 0.0

                max_overlaps, argmax_overlaps = pred_to_gtbox.max(1)
                rm_obj_labels = gt_classes[:, 1][argmax_overlaps]
                rm_obj_labels[max_overlaps < 0.5] = 0
            else:
                rm_obj_labels = None
        else:
            im_inds = rois[:, 0].long().contiguous() + image_offset
            nms_scores = None
            nms_preds = None
            nms_boxes_assign = None
            nms_boxes = None
            box_priors = rois[:, 1:]
            rm_obj_labels = obj_labels
            box_deltas = od_box_deltas
            obj_dists = od_obj_dists

        return Result(
            od_obj_dists=od_obj_dists,
            rm_obj_dists=obj_dists,
            obj_scores=nms_scores,
            obj_preds=nms_preds,
            obj_fmap=obj_fmap,
            od_box_deltas=od_box_deltas,
            rm_box_deltas=box_deltas,
            od_box_targets=bbox_targets,
            rm_box_targets=bbox_targets,
            od_box_priors=od_box_priors,
            rm_box_priors=box_priors,
            boxes_assigned=nms_boxes_assign,
            boxes_all=nms_boxes,
            od_obj_labels=obj_labels,
            rm_obj_labels=rm_obj_labels,
            rpn_scores=rpn_scores,
            rpn_box_deltas=rpn_box_deltas,
            rel_labels=rel_labels,
            im_inds=im_inds,
            fmap=fmap if return_fmap else None,
        )

    def nms_boxes(self, obj_dists, rois, box_deltas, im_sizes):
        """
        Performs NMS on the boxes
        :param obj_dists: [#rois, #classes]
        :param rois: [#rois, 5]
        :param box_deltas: [#rois, #classes, 4]
        :param im_sizes: sizes of images
        :return
            nms_inds [#nms]
            nms_scores [#nms]
            nms_labels [#nms]
            nms_boxes_assign [#nms, 4]
            nms_boxes  [#nms, #classes, 4]. classid=0 is the box prior.
        """
        # Now produce the boxes
        # box deltas is (num_rois, num_classes, 4) but rois is only #(num_rois, 4)
        boxes = bbox_preds(rois[:, None, 1:].expand_as(box_deltas).contiguous().view(-1, 4),
                           box_deltas.view(-1, 4)).view(*box_deltas.size())

        # Clip the boxes and get the best N dets per image.
        inds = rois[:, 0].long().contiguous()
        dets = []
        for i, s, e in enumerate_by_image(inds.data):
            h, w = im_sizes[i, :2]
            boxes[s:e, :, 0].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 1].data.clamp_(min=0, max=h - 1)
            boxes[s:e, :, 2].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 3].data.clamp_(min=0, max=h - 1)
            d_filtered = filter_det(
                F.softmax(obj_dists[s:e], 1), boxes[s:e], start_ind=s,
                nms_filter_duplicates=self.nms_filter_duplicates,
                max_per_img=self.max_per_img,
                thresh=self.thresh,
            )
            if d_filtered is not None:
                dets.append(d_filtered)

        if len(dets) == 0:
            print("nothing was detected", flush=True)
            return None
        nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)]
        twod_inds = nms_inds * boxes.size(1) + nms_labels.data
        nms_boxes_assign = boxes.view(-1, 4)[twod_inds]

        nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, None], boxes[nms_inds][:, 1:]), 1)
        return nms_inds, nms_scores, nms_labels, nms_boxes_assign, nms_boxes, inds[nms_inds]

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if any([x.is_none() for x in outputs]):
            assert not self.training
            return None
        return gather_res(outputs, 0, dim=0)


def filter_det(scores, boxes, start_ind=0, max_per_img=100, thresh=0.001, pre_nms_topn=6000,
               post_nms_topn=300, nms_thresh=0.3, nms_filter_duplicates=True):
    """
    Filters the detections for a single image
    :param scores: [num_rois, num_classes]
    :param boxes: [num_rois, num_classes, 4]. Assumes the boxes have been clamped
    :param max_per_img: Max detections per image
    :param thresh: Threshold for calling it a good box
    :param nms_filter_duplicates: True if we shouldn't allow for mulitple detections of the
           same box (with different labels)
    :return: A numpy concatenated array with up to 100 detections/img [num_im, x1, y1, x2, y2, score, cls]
    """

    valid_cls = (scores[:, 1:].data.max(0)[0] > thresh).nonzero() + 1
    if valid_cls.dim() == 0:
        return None

    nms_mask = scores.data.clone()
    nms_mask.zero_()

    for c_i in valid_cls.squeeze(1).cpu():
        scores_ci = scores.data[:, c_i]
        boxes_ci = boxes.data[:, c_i]

        keep = apply_nms(scores_ci, boxes_ci,
                         pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn,
                         nms_thresh=nms_thresh)
        nms_mask[:, c_i][keep] = 1

    dists_all = Variable(nms_mask * scores.data, volatile=True)

    if nms_filter_duplicates:
        scores_pre, labels_pre = dists_all.data.max(1)
        inds_all = scores_pre.nonzero()
        assert inds_all.dim() != 0
        inds_all = inds_all.squeeze(1)

        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]
    else:
        nz = nms_mask.nonzero()
        assert nz.dim() != 0
        inds_all = nz[:, 0]
        labels_all = nz[:, 1]
        scores_all = scores.data.view(-1)[inds_all * scores.data.size(1) + labels_all]

    # dists_all = dists_all[inds_all]
    # dists_all[:,0] = 1.0-dists_all.sum(1)

    # # Limit to max per image detections
    vs, idx = torch.sort(scores_all, dim=0, descending=True)
    idx = idx[vs > thresh]
    if max_per_img < idx.size(0):
        idx = idx[:max_per_img]

    inds_all = inds_all[idx] + start_ind
    scores_all = Variable(scores_all[idx], volatile=True)
    labels_all = Variable(labels_all[idx], volatile=True)
    # dists_all = dists_all[idx]

    return inds_all, scores_all, labels_all


class RPNHead(nn.Module):
    """
    Serves as the class + box outputs for each level in the FPN.
    """

    def __init__(self, dim=512, input_dim=1024):
        """
        :param aspect_ratios: Aspect ratios for the anchors. NOTE - this can't be changed now
               as it depends on other things in the C code...
        """
        super(RPNHead, self).__init__()

        self.anchor_target_dim = 6
        self.stride = 16

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),  # Tensorflow docs use Relu6, so let's use it too....
            nn.Conv2d(dim, self.anchor_target_dim * self._A,
                      kernel_size=1)
        )

        ans_np = generate_anchors(base_size=ANCHOR_SIZE,
                                  feat_stride=self.stride,
                                  anchor_scales=ANCHOR_SCALES,
                                  anchor_ratios=ANCHOR_RATIOS,
                                  )
        self.register_buffer('anchors', torch.FloatTensor(ans_np))

    @property
    def _A(self):
        return len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)

    def forward(self, fmap):
        """
        Gets the class / noclass predictions over all the scales

        :param fmap: [batch_size, dim, IM_SIZE/16, IM_SIZE/16] featuremap
        :return: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        """
        rez = self._reshape_channels(self.conv(fmap))
        rez = rez.view(rez.size(0), rez.size(1), rez.size(2),
                       self._A, self.anchor_target_dim)
        return rez

    def anchor_preds(self, preds, train_anchor_inds, image_offset):
        """
        Get predictions for the training indices
        :param preds: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        :param train_anchor_inds: [num_train, 4] indices into the predictions
        :return: class_preds: [num_train, 2] array of yes/no
                 box_preds:   [num_train, 4] array of predicted boxes
        """
        assert train_anchor_inds.size(1) == 4
        tai = train_anchor_inds.data.clone()
        tai[:, 0] -= image_offset
        train_regions = gather_nd(preds, tai)

        class_preds = train_regions[:, :2]
        box_preds = train_regions[:, 2:]
        return class_preds, box_preds

    @staticmethod
    def _reshape_channels(x):
        """ [batch_size, channels, h, w] -> [batch_size, h, w, channels] """
        assert x.dim() == 4
        batch_size, nc, h, w = x.size()

        x_t = x.view(batch_size, nc, -1).transpose(1, 2).contiguous()
        x_t = x_t.view(batch_size, h, w, nc)
        return x_t

    def roi_proposals(self, fmap, im_sizes, nms_thresh=0.7, pre_nms_topn=12000, post_nms_topn=2000):
        """
        :param fmap: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        :param im_sizes:        [batch_size, 3] numpy array of (h, w, scale)
        :return: ROIS: shape [a <=post_nms_topn, 5] array of ROIS.
        """
        class_fmap = fmap[:, :, :, :, :2].contiguous()

        # GET THE GOOD BOXES AYY LMAO :')
        class_preds = F.softmax(class_fmap, 4)[..., 1].data.contiguous()

        box_fmap = fmap[:, :, :, :, 2:].data.contiguous()

        anchor_stacked = torch.cat([self.anchors[None]] * fmap.size(0), 0)
        box_preds = bbox_preds(anchor_stacked.view(-1, 4), box_fmap.view(-1, 4)).view(
            *box_fmap.size())

        for i, (h, w, scale) in enumerate(im_sizes):
            # Zero out all the bad boxes h, w, A, 4
            h_end = int(h) // self.stride
            w_end = int(w) // self.stride
            if h_end < class_preds.size(1):
                class_preds[i, h_end:] = -0.01
            if w_end < class_preds.size(2):
                class_preds[i, :, w_end:] = -0.01

            # and clamp the others
            box_preds[i, :, :, :, 0].clamp_(min=0, max=w - 1)
            box_preds[i, :, :, :, 1].clamp_(min=0, max=h - 1)
            box_preds[i, :, :, :, 2].clamp_(min=0, max=w - 1)
            box_preds[i, :, :, :, 3].clamp_(min=0, max=h - 1)

        sizes = center_size(box_preds.view(-1, 4))
        class_preds.view(-1)[(sizes[:, 2] < 4) | (sizes[:, 3] < 4)] = -0.01
        return filter_roi_proposals(box_preds.view(-1, 4), class_preds.view(-1),
                                    boxes_per_im=np.array([np.prod(box_preds.size()[1:-1])] * fmap.size(0)),
                                    nms_thresh=nms_thresh,
                                    pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn)


def filter_roi_proposals(box_preds, class_preds, boxes_per_im, nms_thresh=0.7, pre_nms_topn=12000, post_nms_topn=2000):
    inds, im_per = apply_nms(
        class_preds,
        box_preds,
        pre_nms_topn=pre_nms_topn,
        post_nms_topn=post_nms_topn,
        boxes_per_im=boxes_per_im,
        nms_thresh=nms_thresh,
    )
    img_inds = torch.cat([val * torch.ones(i) for val, i in enumerate(im_per)], 0).cuda(
        box_preds.get_device())
    rois = torch.cat((img_inds[:, None], box_preds[inds]), 1)
    return rois


def load_resnet():
    model = resnet101(pretrained=True)
    del model.layer4
    del model.avgpool
    del model.fc
    return model


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    model = vgg16(pretrained=pretrained)
    del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    if not use_dropout:
        del model.classifier._modules['5']  # Get rid of dropout
        if not use_relu:
            del model.classifier._modules['4']  # Get rid of relu activation
            if not use_linear:
                del model.classifier._modules['3']  # Get rid of linear layer
    return model
