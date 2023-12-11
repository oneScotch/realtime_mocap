# yapf: disable
import cv2
import numpy as np
from xrprimer.transform.image.color import bgr2rgb

# yapf: enable
SIZE = 256
MEAN = np.array([0.485, 0.456,
                 0.406]).reshape(3, 1, 1).repeat(SIZE, 1).repeat(SIZE, 2)
STD = np.array([0.229, 0.224,
                0.225]).reshape(3, 1, 1).repeat(SIZE, 1).repeat(SIZE, 2)


def pad2square(img, color=None):
    if img.shape[0] > img.shape[1]:
        W = img.shape[0] - img.shape[1]
    else:
        W = img.shape[1] - img.shape[0]
    W1 = int(W / 2)
    W2 = W - W1
    if color is None:
        if img.shape[2] == 3:
            color = (0, 0, 0)
        else:
            color = 0
    if img.shape[0] > img.shape[1]:
        return cv2.copyMakeBorder(
            img, 0, 0, W1, W2, cv2.BORDER_CONSTANT, value=color)
    else:
        return cv2.copyMakeBorder(
            img, W1, W2, 0, 0, cv2.BORDER_CONSTANT, value=color)


def transform_img_intaghand(img):
    img = pad2square(img)
    img = cv2.resize(img, (SIZE, SIZE))
    img = bgr2rgb(img, -1)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img


def pad_resize_img_intaghand(img):
    img = pad2square(img)
    img = cv2.resize(img, (SIZE, SIZE))
    return img


def normalize_img_intaghand(img):
    img = bgr2rgb(img, -1)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img
