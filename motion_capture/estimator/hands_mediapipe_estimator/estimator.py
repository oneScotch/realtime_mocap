import numpy as np
from typing import Union
from xrmocap.data_structure.keypoints import Keypoints
from xrprimer.transform.image.color import bgr2rgb

from ..base_estimator import BaseEstimator
from ..detect_boost.builder import build_detect_boost

try:
    import mediapipe as mp
    has_mediapipe = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mediapipe = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class HandsMediapipeEstimator(BaseEstimator):

    def __init__(self,
                 hand_model_complexity=1,
                 hand_max_num_hands=2,
                 hand_min_detection_confidence=0.7,
                 hand_min_tracking_confidence=0.5,
                 detect_boost: Union[dict, None] = None,
                 logger=None) -> None:
        BaseEstimator.__init__(self, logger=logger)
        if not has_mediapipe:
            self.logger.error(import_exception)
            raise ImportError
        self.hands_mp = mp.solutions.hands.Hands(
            model_complexity=hand_model_complexity,
            max_num_hands=hand_max_num_hands,
            min_detection_confidence=hand_min_detection_confidence,
            min_tracking_confidence=hand_min_tracking_confidence,
        )
        if detect_boost is None:
            self.detect_boost = detect_boost
        else:
            self.detect_boost = build_detect_boost(detect_boost)

    def forward(self, img_arr, k4a_pose, **kwargs):
        ret_dict = dict()
        ret_dict.update(kwargs)
        img = bgr2rgb(img_arr.copy())
        if self.detect_boost is not None:
            img = self.detect_boost.get_image(img, k4a_pose)
        hands_result_mp = self.hands_mp.process(img)

        hands_keypoints2d = dict(left=None, right=None)
        hands_flag = []
        if hands_result_mp.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(
                    hands_result_mp.multi_hand_landmarks):
                kps_list = [[landmark.x, landmark.y, 1]
                            for landmark in hand_landmarks.landmark]
                # side: left or right
                side = str.lower(hands_result_mp.multi_handedness[idx].
                                 classification[0].label)
                hands_flag.append(side)
                kps2d = np.expand_dims(np.array(kps_list), (0, 1))
                kps2d[..., 0] *= img.shape[1]
                kps2d[..., 1] *= img.shape[0]
                if self.detect_boost is not None:
                    kps2d = self.detect_boost.get_points(kps2d)
                # Keypoints class causes an error in mp queue
                hands_keypoints2d[side] = dict(
                    Keypoints(
                        dtype='numpy',
                        kps=kps2d,
                        mask=np.ones_like(kps2d[..., 0]),
                        convention=f'mano_{side}_reorder'))
        # add visible args to ret_dict
        ret_dict['k4a_pose'] = k4a_pose
        ret_dict['img_arr'] = img_arr
        ret_dict['hands_keypoints2d'] = hands_keypoints2d
        return ret_dict
