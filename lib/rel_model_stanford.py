"""
Let's get the relationships yo
"""

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.surgery import filter_dets
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.pytorch_misc import arange
from lib.object_detector import filter_det
from lib.rel_model import RelModel

MODES = ('sgdet', 'sgcls', 'predcls')

SIZE=512

class RelModelStanford(RelModel):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                 use_resnet=False, use_proposals=False, **kwargs):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(RelModelStanford, self).__init__(classes, rel_classes, mode=mode, num_gpus=num_gpus,
                                               require_overlap_det=require_overlap_det,
                                               use_resnet=use_resnet,
                                               nl_obj=0, nl_edge=0, use_proposals=use_proposals, thresh=0.01,
                                               pooling_dim=4096)

        del self.context
        del self.post_lstm
        del self.post_emb

        self.rel_fc = nn.Linear(SIZE, self.num_rels)
        self.obj_fc = nn.Linear(SIZE, self.num_classes)

        self.obj_unary = nn.Linear(self.obj_dim, SIZE)
        self.edge_unary = nn.Linear(4096, SIZE)


        self.edge_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)
        self.node_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)

        self.n_iter = 3

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())

        self.in_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())

    def message_pass(self, rel_rep, obj_rep, rel_inds):
        """

        :param rel_rep: [num_rel, fc]
        :param obj_rep: [num_obj, fc]
        :param rel_inds: [num_rel, 2] of the valid relationships
        :return: object prediction [num_obj, 151], bbox_prediction [num_obj, 151*4] 
                and rel prediction [num_rel, 51]
        """
        # [num_obj, num_rel] with binary!
        numer = torch.arange(0, rel_inds.size(0)).long().cuda(rel_inds.get_device())

        objs_to_outrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_outrels.view(-1)[rel_inds[:, 0] * rel_rep.size(0) + numer] = 1
        objs_to_outrels = Variable(objs_to_outrels)

        objs_to_inrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_inrels.view(-1)[rel_inds[:, 1] * rel_rep.size(0) + numer] = 1
        objs_to_inrels = Variable(objs_to_inrels)

        hx_rel = Variable(rel_rep.data.new(rel_rep.size(0), SIZE).zero_(), requires_grad=False)
        hx_obj = Variable(obj_rep.data.new(obj_rep.size(0), SIZE).zero_(), requires_grad=False)

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(3):
            # compute edge context
            sub_vert = vert_factor[i][rel_inds[:, 0]]
            obj_vert = vert_factor[i][rel_inds[:, 1]]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * \
                      edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[
                i]

            vert_ctx = objs_to_outrels @ pre_out + objs_to_inrels @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        # woohoo! done
        return self.obj_fc(vert_factor[-1]), self.rel_fc(edge_factor[-1])
               # self.box_fc(vert_factor[-1]).view(-1, self.num_classes, 4), \
               # self.rel_fc(edge_factor[-1])

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
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True, num_sample_per_gt=1)
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        visual_rep = self.visual_rep(result.fmap, rois, rel_inds[:, 1:])

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # Now do the approximation WHEREVER THERES A VALID RELATIONSHIP.
        result.rm_obj_dists, result.rel_dists = self.message_pass(
            F.relu(self.edge_unary(visual_rep)), self.obj_unary(result.obj_fmap), rel_inds[:, 1:])

        # result.box_deltas_update = box_deltas

        if self.training:
            return result

        # Decode here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mode == 'predcls':
            # Hack to get the GT object labels
            result.obj_scores = result.rm_obj_dists.data.new(gt_classes.size(0)).fill_(1)
            result.obj_preds = gt_classes.data[:, 1]
        elif self.mode == 'sgdet':
            order, obj_scores, obj_preds= filter_det(F.softmax(result.rm_obj_dists),
                                                              result.boxes_all,
                                                              start_ind=0,
                                                              max_per_img=100,
                                                              thresh=0.00,
                                                              pre_nms_topn=6000,
                                                              post_nms_topn=300,
                                                              nms_thresh=0.3,
                                                              nms_filter_duplicates=True)
            idx, perm = torch.sort(order)
            result.obj_preds = rel_inds.new(result.rm_obj_dists.size(0)).fill_(1)
            result.obj_scores = result.rm_obj_dists.data.new(result.rm_obj_dists.size(0)).fill_(0)
            result.obj_scores[idx] = obj_scores.data[perm]
            result.obj_preds[idx] = obj_preds.data[perm]
        else:
            scores_nz = F.softmax(result.rm_obj_dists).data
            scores_nz[:, 0] = 0.0
            result.obj_scores, score_ord = scores_nz[:, 1:].sort(dim=1, descending=True)
            result.obj_preds = score_ord[:,0] + 1
            result.obj_scores = result.obj_scores[:,0]

        result.obj_preds = Variable(result.obj_preds)
        result.obj_scores = Variable(result.obj_scores)

        # Set result's bounding boxes to be size
        # [num_boxes, topk, 4] instead of considering every single object assignment.
        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data

        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors
        rel_rep = F.softmax(result.rel_dists)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

