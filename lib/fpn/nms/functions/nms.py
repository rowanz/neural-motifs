# Le code for doing NMS
import torch
import numpy as np
from .._ext import nms


def apply_nms(scores, boxes,  pre_nms_topn=12000, post_nms_topn=2000, boxes_per_im=None,
              nms_thresh=0.7):
    """
    Note - this function is non-differentiable so everything is assumed to be a tensor, not
    a variable.
        """
    just_inds = boxes_per_im is None
    if boxes_per_im is None:
        boxes_per_im = [boxes.size(0)]


    s = 0
    keep = []
    im_per = []
    for bpi in boxes_per_im:
        e = s + int(bpi)
        keep_im = _nms_single_im(scores[s:e], boxes[s:e], pre_nms_topn, post_nms_topn, nms_thresh)
        keep.append(keep_im + s)
        im_per.append(keep_im.size(0))

        s = e

    inds = torch.cat(keep, 0)
    if just_inds:
        return inds
    return inds, im_per


def _nms_single_im(scores, boxes,  pre_nms_topn=12000, post_nms_topn=2000, nms_thresh=0.7):
    keep = torch.IntTensor(scores.size(0))
    vs, idx = torch.sort(scores, dim=0, descending=True)
    if idx.size(0) > pre_nms_topn:
        idx = idx[:pre_nms_topn]
    boxes_sorted = boxes[idx].contiguous()
    num_out = nms.nms_apply(keep, boxes_sorted, nms_thresh)
    num_out = min(num_out, post_nms_topn)
    keep = keep[:num_out].long()
    keep = idx[keep.cuda(scores.get_device())]
    return keep
