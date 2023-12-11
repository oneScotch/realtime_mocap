# yapf: disable
import numpy as np

# yapf: enable
SIZE = 224
MEAN = np.array([0.485, 0.456,
                 0.406]).reshape(3, 1, 1).repeat(SIZE, 1).repeat(SIZE, 2)
STD = np.array([0.229, 0.224,
                0.225]).reshape(3, 1, 1).repeat(SIZE, 1).repeat(SIZE, 2)


def normalize_img_frankbody(img):
    img = img[:, :, ::-1].copy()
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.expand_dims(img, axis=0)
    return img
