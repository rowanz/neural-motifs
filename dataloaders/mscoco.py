from config import COCO_PATH, IM_SCALE, BOX_SCALE
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from lib.fpn.anchor_targets import anchor_target_layer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, RandomOrder, Hue, random_crop
import numpy as np
from dataloaders.blob import Blob
import torch

class CocoDetection(Dataset):
    """
    Adapted from the torchvision code
    """

    def __init__(self, mode):
        """
        :param mode: train2014 or val2014
        """
        self.mode = mode
        self.root = os.path.join(COCO_PATH, mode)
        self.ann_file = os.path.join(COCO_PATH, 'annotations', 'instances_{}.json'.format(mode))
        self.coco = COCO(self.ann_file)
        self.ids = [k for k in self.coco.imgs.keys() if len(self.coco.imgToAnns[k]) > 0]


        tform = []
        if self.is_train:
             tform.append(RandomOrder([
                 Grayscale(),
                 Brightness(),
                 Contrast(),
                 Sharpness(),
                 Hue(),
             ]))

        tform += [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.transform_pipeline = Compose(tform)
        self.ind_to_classes = ['__background__'] + [v['name'] for k, v in self.coco.cats.items()]
        # COCO inds are weird (84 inds in total but a bunch of numbers are skipped)
        self.id_to_ind = {coco_id:(ind+1) for ind, coco_id in enumerate(self.coco.cats.keys())}
        self.id_to_ind[0] = 0

        self.ind_to_id = {x:y for y,x in self.id_to_ind.items()}

    @property
    def is_train(self):
        return self.mode.startswith('train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns: entry dict
        """
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image_unpadded = Image.open(os.path.join(self.root, path)).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        gt_classes = np.array([self.id_to_ind[x['category_id']] for x in anns], dtype=np.int64)

        if np.any(gt_classes >= len(self.ind_to_classes)):
            raise ValueError("OH NO {}".format(index))

        if len(anns) == 0:
            raise ValueError("Annotations should not be empty")
        #     gt_boxes = np.array((0, 4), dtype=np.float32)
        # else:
        gt_boxes = np.array([x['bbox'] for x in anns], dtype=np.float32)

        if np.any(gt_boxes[:, [0,1]] < 0):
            raise ValueError("GT boxes empty columns")
        if np.any(gt_boxes[:, [2,3]] < 0):
            raise ValueError("GT boxes empty h/w")
        gt_boxes[:, [2, 3]] += gt_boxes[:, [0, 1]]

        # Rescale so that the boxes are at BOX_SCALE
        if self.is_train:
            image_unpadded, gt_boxes = random_crop(image_unpadded,
                                                   gt_boxes * BOX_SCALE / max(image_unpadded.size),
                                                   BOX_SCALE,
                                                   round_boxes=False,
                                                   )
        else:
            # Seems a bit silly because we won't be using GT boxes then but whatever
            gt_boxes = gt_boxes * BOX_SCALE / max(image_unpadded.size)
        w, h = image_unpadded.size
        box_scale_factor = BOX_SCALE / max(w, h)

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w*img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h*img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'scale': IM_SCALE / BOX_SCALE,
            'index': index,
            'image_id': img_id,
            'flipped': flipped,
            'fn': path,
        }

        return entry

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train2014', *args, **kwargs)
        val = cls('val2014', *args, **kwargs)
        return train, val

    def __len__(self):
        return len(self.ids)


def coco_collate(data, num_gpus=3, is_train=False):
    blob = Blob(mode='det', is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class CocoDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """
    # def __iter__(self):
    #     for x in super(CocoDataLoader, self).__iter__():
    #         if isinstance(x, tuple) or isinstance(x, list):
    #             yield tuple(y.cuda(async=True) if hasattr(y, 'cuda') else y for y in x)
    #         else:
    #             yield x.cuda(async=True)

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, **kwargs):
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size*num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: coco_collate(x, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size*num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: coco_collate(x, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load


if __name__ == '__main__':
    train, val = CocoDetection.splits()
    gtbox = train[0]['gt_boxes']
    img_size = train[0]['img_size']
    anchor_strides, labels, bbox_targets = anchor_target_layer(gtbox, img_size)
