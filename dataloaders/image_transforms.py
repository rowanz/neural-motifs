# Some image transforms

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
from random import randint
# All of these need to be called on PIL imagez

class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=(int(0.485 * 256), int(0.456 * 256), int(0.406 * 256)))
        return img_padded


class Grayscale(object):
    """
    Converts to grayscale (not always, sometimes).
    """
    def __call__(self, img):
        factor = np.sqrt(np.sqrt(np.random.rand(1)))
        # print("gray {}".format(factor))
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)


class Brightness(object):
    """
    Converts to grayscale (not always, sometimes).
    """
    def __call__(self, img):
        factor = np.random.randn(1)/6+1
        factor = min(max(factor, 0.5), 1.5)
        # print("brightness {}".format(factor))

        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)


class Contrast(object):
    """
    Converts to grayscale (not always, sometimes).
    """
    def __call__(self, img):
        factor = np.random.randn(1)/8+1.0
        factor = min(max(factor, 0.5), 1.5)
        # print("contrast {}".format(factor))

        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)


class Hue(object):
    """
    Converts to grayscale
    """
    def __call__(self, img):
        # 30 seems good
        factor = int(np.random.randn(1)*8)
        factor = min(max(factor, -30), 30)
        factor = np.array(factor, dtype=np.uint8)

        hsv = np.array(img.convert('HSV'))
        hsv[:,:,0] += factor
        new_img = Image.fromarray(hsv, 'HSV').convert('RGB')

        return new_img


class Sharpness(object):
    """
    Converts to grayscale
    """
    def __call__(self, img):
        factor = 1.0 + np.random.randn(1)/5
        # print("sharpness {}".format(factor))
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)


def random_crop(img, boxes, box_scale, round_boxes=True, max_crop_fraction=0.1):
    """
    Randomly crops the image
    :param img: PIL image
    :param boxes: Ground truth boxes
    :param box_scale: This is the scale that the boxes are at (e.g. 1024 wide). We'll preserve that ratio
    :param round_boxes: Set this to true if we're going to round the boxes to ints
    :return: Cropped image, new boxes
    """

    w, h = img.size

    max_crop_w = int(w*max_crop_fraction)
    max_crop_h = int(h*max_crop_fraction)
    boxes_scaled = boxes * max(w,h) / box_scale
    max_to_crop_top = min(int(boxes_scaled[:, 1].min()), max_crop_h)
    max_to_crop_left = min(int(boxes_scaled[:, 0].min()), max_crop_w)
    max_to_crop_right = min(int(w - boxes_scaled[:, 2].max()), max_crop_w)
    max_to_crop_bottom = min(int(h - boxes_scaled[:, 3].max()), max_crop_h)

    crop_top = randint(0, max(max_to_crop_top, 0))
    crop_left = randint(0, max(max_to_crop_left, 0))
    crop_right = randint(0, max(max_to_crop_right, 0))
    crop_bottom = randint(0, max(max_to_crop_bottom, 0))
    img_cropped = img.crop((crop_left, crop_top, w - crop_right, h - crop_bottom))

    new_boxes = box_scale / max(img_cropped.size) * np.column_stack(
        (boxes_scaled[:,0]-crop_left, boxes_scaled[:,1]-crop_top, boxes_scaled[:,2]-crop_left, boxes_scaled[:,3]-crop_top))

    if round_boxes:
        new_boxes = np.round(new_boxes).astype(np.int32)
    return img_cropped, new_boxes


class RandomOrder(object):
    """ Composes several transforms together in random order - or not at all!
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        num_to_pick = np.random.choice(len(self.transforms))
        if num_to_pick == 0:
            return img

        order = np.random.choice(len(self.transforms), size=num_to_pick, replace=False)
        for i in order:
            img = self.transforms[i](img)
        return img