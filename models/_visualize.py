"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.rel_model import RelModel
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )
detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(train.ind_to_classes) + 1)


def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image


font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)


def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)

    evaluator[conf.mode].print_stats()


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
    det_res = detector[b]
    # if conf.num_gpus == 1:
    #     det_res = [det_res]
    assert conf.num_gpus == 1
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res

    gt_entry = {
        'gt_classes': val.gt_classes[batch_num].copy(),
        'gt_relations': val.relationships[batch_num].copy(),
        'gt_boxes': val.gt_boxes[batch_num].copy(),
    }
    # gt_entry = {'gt_classes': gtc[i], 'gt_relations': gtr[i], 'gt_boxes': gtb[i]}
    assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
    # assert np.all(rels_i[:, 2] > 0)

    pred_entry = {
        'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
        'pred_classes': objs_i,
        'pred_rel_inds': rels_i,
        'obj_scores': obj_scores_i,
        'rel_scores': pred_scores_i,
    }
    pred_to_gt, pred_5ples, rel_scores = evaluator[conf.mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    # SET RECALL THRESHOLD HERE
    pred_to_gt = pred_to_gt[:20]
    pred_5ples = pred_5ples[:20]

    # Get a list of objects that match, and GT objects that dont
    objs_match = (bbox_overlaps(pred_entry['pred_boxes'], gt_entry['gt_boxes']) >= 0.5) & (
            objs_i[:, None] == gt_entry['gt_classes'][None]
    )
    objs_matched = objs_match.any(1)

    has_seen = defaultdict(int)
    has_seen_gt = defaultdict(int)
    pred_ind2name = {}
    gt_ind2name = {}
    edges = {}
    missededges = {}
    badedges = {}

    if val.filenames[batch_num].startswith('2343676'):
        import ipdb
        ipdb.set_trace()

    def query_pred(pred_ind):
        if pred_ind not in pred_ind2name:
            has_seen[objs_i[pred_ind]] += 1
            pred_ind2name[pred_ind] = '{}-{}'.format(train.ind_to_classes[objs_i[pred_ind]],
                                                     has_seen[objs_i[pred_ind]])
        return pred_ind2name[pred_ind]

    def query_gt(gt_ind):
        gt_cls = gt_entry['gt_classes'][gt_ind]
        if gt_ind not in gt_ind2name:
            has_seen_gt[gt_cls] += 1
            gt_ind2name[gt_ind] = '{}-GT{}'.format(train.ind_to_classes[gt_cls], has_seen_gt[gt_cls])
        return gt_ind2name[gt_ind]

    matching_pred5ples = pred_5ples[np.array([len(x) > 0 for x in pred_to_gt])]
    for fiveple in matching_pred5ples:
        head_name = query_pred(fiveple[0])
        tail_name = query_pred(fiveple[1])

        edges[(head_name, tail_name)] = train.ind_to_predicates[fiveple[4]]

    gt_5ples = np.column_stack((gt_entry['gt_relations'][:, :2],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 0]],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 1]],
                                gt_entry['gt_relations'][:, 2],
                                ))
    has_match = reduce(np.union1d, pred_to_gt)
    for gt in gt_5ples[np.setdiff1d(np.arange(gt_5ples.shape[0]), has_match)]:
        # Head and tail
        namez = []
        for i in range(2):
            matching_obj = np.where(objs_match[:, gt[i]])[0]
            if matching_obj.size > 0:
                name = query_pred(matching_obj[0])
            else:
                name = query_gt(gt[i])
            namez.append(name)

        missededges[tuple(namez)] = train.ind_to_predicates[gt[4]]

    for fiveple in pred_5ples[np.setdiff1d(np.arange(pred_5ples.shape[0]), matching_pred5ples)]:

        if fiveple[0] in pred_ind2name:
            if fiveple[1] in pred_ind2name:
                badedges[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = train.ind_to_predicates[fiveple[4]]

    theimg = load_unscaled(val.filenames[batch_num])
    theimg2 = theimg.copy()
    draw2 = ImageDraw.Draw(theimg2)

    # Fix the names

    for pred_ind in pred_ind2name.keys():
        draw2 = draw_box(draw2, pred_entry['pred_boxes'][pred_ind],
                         cls_ind=objs_i[pred_ind],
                         text_str=pred_ind2name[pred_ind])
    for gt_ind in gt_ind2name.keys():
        draw2 = draw_box(draw2, gt_entry['gt_boxes'][gt_ind],
                         cls_ind=gt_entry['gt_classes'][gt_ind],
                         text_str=gt_ind2name[gt_ind])

    recall = int(100 * len(reduce(np.union1d, pred_to_gt)) / gt_entry['gt_relations'].shape[0])

    id = '{}-{}'.format(val.filenames[batch_num].split('/')[-1][:-4], recall)
    pathname = os.path.join('qualitative', id)
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    theimg.save(os.path.join(pathname, 'img.jpg'), quality=100, subsampling=0)
    theimg2.save(os.path.join(pathname, 'imgbox.jpg'), quality=100, subsampling=0)

    with open(os.path.join(pathname, 'shit.txt'), 'w') as f:
        f.write('good:\n')
        for (o1, o2), p in edges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        f.write('fn:\n')
        for (o1, o2), p in missededges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        f.write('shit:\n')
        for (o1, o2), p in badedges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))


mAp = val_epoch()
