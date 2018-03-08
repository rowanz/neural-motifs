"""
Data blob, hopefully to make collating less painful and MGPU training possible
"""
from lib.fpn.anchor_targets import anchor_target_layer
import numpy as np
import torch
from torch.autograd import Variable


class Blob(object):
    def __init__(self, mode='det', is_train=False, num_gpus=1, primary_gpu=0, batch_size_per_gpu=3):
        """
        Initializes an empty Blob object.
        :param mode: 'det' for detection and 'rel' for det+relationship
        :param is_train: True if it's training
        """
        assert mode in ('det', 'rel')
        assert num_gpus >= 1
        self.mode = mode
        self.is_train = is_train
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = batch_size_per_gpu
        self.primary_gpu = primary_gpu

        self.imgs = []  # [num_images, 3, IM_SCALE, IM_SCALE] array
        self.im_sizes = []  # [num_images, 4] array of (h, w, scale, num_valid_anchors)
        self.all_anchor_inds = []  # [all_anchors, 2] array of (img_ind, anchor_idx). Only has valid
        # boxes (meaning some are gonna get cut out)
        self.all_anchors = []  # [num_im, IM_SCALE/4, IM_SCALE/4, num_anchors, 4] shapes. Anchors outside get squashed
                               # to 0
        self.gt_boxes = []  # [num_gt, 4] boxes
        self.gt_classes = []  # [num_gt,2] array of img_ind, class
        self.gt_rels = []  # [num_rels, 3]. Each row is (gtbox0, gtbox1, rel).

        self.gt_sents = []
        self.gt_nodes = []
        self.sent_lengths = []

        self.train_anchor_labels = []  # [train_anchors, 5] array of (img_ind, h, w, A, labels)
        self.train_anchors = []  # [train_anchors, 8] shapes with anchor, target

        self.train_anchor_inds = None  # This will be split into GPUs, just (img_ind, h, w, A).

        self.batch_size = None
        self.gt_box_chunks = None
        self.anchor_chunks = None
        self.train_chunks = None
        self.proposal_chunks = None
        self.proposals = []

    @property
    def is_flickr(self):
        return self.mode == 'flickr'

    @property
    def is_rel(self):
        return self.mode == 'rel'

    @property
    def volatile(self):
        return not self.is_train

    def append(self, d):
        """
        Adds a single image to the blob
        :param datom:
        :return:
        """
        i = len(self.imgs)
        self.imgs.append(d['img'])

        h, w, scale = d['img_size']

        # all anchors
        self.im_sizes.append((h, w, scale))

        gt_boxes_ = d['gt_boxes'].astype(np.float32) * d['scale']
        self.gt_boxes.append(gt_boxes_)

        self.gt_classes.append(np.column_stack((
            i * np.ones(d['gt_classes'].shape[0], dtype=np.int64),
            d['gt_classes'],
        )))

        # Add relationship info
        if self.is_rel:
            self.gt_rels.append(np.column_stack((
                i * np.ones(d['gt_relations'].shape[0], dtype=np.int64),
                d['gt_relations'])))

        # Augment with anchor targets
        if self.is_train:
            train_anchors_, train_anchor_inds_, train_anchor_targets_, train_anchor_labels_ = \
                anchor_target_layer(gt_boxes_, (h, w))

            self.train_anchors.append(np.hstack((train_anchors_, train_anchor_targets_)))

            self.train_anchor_labels.append(np.column_stack((
                i * np.ones(train_anchor_inds_.shape[0], dtype=np.int64),
                train_anchor_inds_,
                train_anchor_labels_,
            )))

        if 'proposals' in d:
            self.proposals.append(np.column_stack((i * np.ones(d['proposals'].shape[0], dtype=np.float32),
                                                   d['scale'] * d['proposals'].astype(np.float32))))



    def _chunkize(self, datom, tensor=torch.LongTensor):
        """
        Turn data list into chunks, one per GPU
        :param datom: List of lists of numpy arrays that will be concatenated.
        :return:
        """
        chunk_sizes = [0] * self.num_gpus
        for i in range(self.num_gpus):
            for j in range(self.batch_size_per_gpu):
                chunk_sizes[i] += datom[i * self.batch_size_per_gpu + j].shape[0]
        return Variable(tensor(np.concatenate(datom, 0)), volatile=self.volatile), chunk_sizes

    def reduce(self):
        """ Merges all the detections into flat lists + numbers of how many are in each"""
        if len(self.imgs) != self.batch_size_per_gpu * self.num_gpus:
            raise ValueError("Wrong batch size? imgs len {} bsize/gpu {} numgpus {}".format(
                len(self.imgs), self.batch_size_per_gpu, self.num_gpus
            ))

        self.imgs = Variable(torch.stack(self.imgs, 0), volatile=self.volatile)
        self.im_sizes = np.stack(self.im_sizes).reshape(
            (self.num_gpus, self.batch_size_per_gpu, 3))

        if self.is_rel:
            self.gt_rels, self.gt_rel_chunks = self._chunkize(self.gt_rels)

        self.gt_boxes, self.gt_box_chunks = self._chunkize(self.gt_boxes, tensor=torch.FloatTensor)
        self.gt_classes, _ = self._chunkize(self.gt_classes)
        if self.is_train:
            self.train_anchor_labels, self.train_chunks = self._chunkize(self.train_anchor_labels)
            self.train_anchors, _ = self._chunkize(self.train_anchors, tensor=torch.FloatTensor)
            self.train_anchor_inds = self.train_anchor_labels[:, :-1].contiguous()

        if len(self.proposals) != 0:
            self.proposals, self.proposal_chunks = self._chunkize(self.proposals, tensor=torch.FloatTensor)



    def _scatter(self, x, chunk_sizes, dim=0):
        """ Helper function"""
        if self.num_gpus == 1:
            return x.cuda(self.primary_gpu, async=True)
        return torch.nn.parallel.scatter_gather.Scatter.apply(
            list(range(self.num_gpus)), chunk_sizes, dim, x)

    def scatter(self):
        """ Assigns everything to the GPUs"""
        self.imgs = self._scatter(self.imgs, [self.batch_size_per_gpu] * self.num_gpus)

        self.gt_classes_primary = self.gt_classes.cuda(self.primary_gpu, async=True)
        self.gt_boxes_primary = self.gt_boxes.cuda(self.primary_gpu, async=True)

        # Predcls might need these
        self.gt_classes = self._scatter(self.gt_classes, self.gt_box_chunks)
        self.gt_boxes = self._scatter(self.gt_boxes, self.gt_box_chunks)

        if self.is_train:

            self.train_anchor_inds = self._scatter(self.train_anchor_inds,
                                                   self.train_chunks)
            self.train_anchor_labels = self.train_anchor_labels.cuda(self.primary_gpu, async=True)
            self.train_anchors = self.train_anchors.cuda(self.primary_gpu, async=True)

            if self.is_rel:
                self.gt_rels = self._scatter(self.gt_rels, self.gt_rel_chunks)
        else:
            if self.is_rel:
                self.gt_rels = self.gt_rels.cuda(self.primary_gpu, async=True)

        if self.proposal_chunks is not None:
            self.proposals = self._scatter(self.proposals, self.proposal_chunks)

    def __getitem__(self, index):
        """
        Returns a tuple containing data
        :param index: Which GPU we're on, or 0 if no GPUs
        :return: If training:
        (image, im_size, img_start_ind, anchor_inds, anchors, gt_boxes, gt_classes, 
        train_anchor_inds)
        test:
        (image, im_size, img_start_ind, anchor_inds, anchors)
        """
        if index not in list(range(self.num_gpus)):
            raise ValueError("Out of bounds with index {} and {} gpus".format(index, self.num_gpus))

        if self.is_rel:
            rels = self.gt_rels
            if index > 0 or self.num_gpus != 1:
                rels_i = rels[index] if self.is_rel else None
        elif self.is_flickr:
            rels = (self.gt_sents, self.gt_nodes)
            if index > 0 or self.num_gpus != 1:
                rels_i = (self.gt_sents[index], self.gt_nodes[index])
        else:
            rels = None
            rels_i = None

        if self.proposal_chunks is None:
            proposals = None
        else:
            proposals = self.proposals

        if index == 0 and self.num_gpus == 1:
            image_offset = 0
            if self.is_train:
                return (self.imgs, self.im_sizes[0], image_offset,
                        self.gt_boxes, self.gt_classes, rels, proposals, self.train_anchor_inds)
            return self.imgs, self.im_sizes[0], image_offset, self.gt_boxes, self.gt_classes, rels, proposals

        # Otherwise proposals is None
        assert proposals is None

        image_offset = self.batch_size_per_gpu * index
        # TODO: Return a namedtuple
        if self.is_train:
            return (
            self.imgs[index], self.im_sizes[index], image_offset,
            self.gt_boxes[index], self.gt_classes[index], rels_i, None, self.train_anchor_inds[index])
        return (self.imgs[index], self.im_sizes[index], image_offset,
                self.gt_boxes[index], self.gt_classes[index], rels_i, None)

