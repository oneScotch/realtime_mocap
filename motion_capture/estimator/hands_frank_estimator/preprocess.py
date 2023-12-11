# yapf: disable
import numpy as np

# yapf: enable
SIZE = 224
MEAN = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1).repeat(SIZE,
                                                         1).repeat(SIZE, 2)
STD = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1).repeat(SIZE,
                                                        1).repeat(SIZE, 2)


def normalize_img_frankmocap(img):
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img


def get_input_batch_frank(cropped_imgs, bbox_dict):
    input_np_list = []
    for side in ('left', 'right'):
        img = cropped_imgs[side]
        if side == 'left':
            img = np.flip(img, axis=1)
        input_np = normalize_img_frankmocap(img)
        input_np = np.expand_dims(input_np, 0)
        input_np_list.append(input_np)
    input_np = np.concatenate(input_np_list, axis=0)
    return input_np
