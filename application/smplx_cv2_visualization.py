import cv2
import numpy as np
import torch
from typing import Union
from xrmocap.data_structure.body_model import SMPLXData
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.ops.projection.opencv_projector import OpencvProjector

from .base_application import BaseApplication
from .body_model.builder import build_body_model


class SmplxCv2Visualization(BaseApplication):

    def __init__(self,
                 device: str,
                 body_model: dict,
                 fisheye_param_path: str,
                 window_size: Union[int, None] = 1280,
                 tmp_img_path: Union[str, None] = None,
                 logger=None) -> None:
        BaseApplication.__init__(self, logger=logger)
        self.device = device
        self.fisheye_param = FisheyeCameraParameter.fromfile(
            fisheye_param_path)
        self.projector = OpencvProjector(
            camera_parameters=[self.fisheye_param], logger=self.logger)
        # set device before build always fails
        # body_model['device'] = device
        body_model = build_body_model(body_model)
        self.body_model = body_model.to(self.device)
        self.window_size = window_size
        self.new_size = None
        self.tmp_img_path = tmp_img_path

    def forward(self, smplx_data: SMPLXData, img: np.ndarray = None, **kwargs):
        with torch.no_grad():
            param_dict = smplx_data.to_tensor_dict(device=self.device)
            mesh_result = self.body_model(
                betas=param_dict['betas'],
                global_orient=param_dict['global_orient'],
                body_pose=param_dict['body_pose'],
                left_hand_pose=param_dict['left_hand_pose'].reshape(1, 45),
                right_hand_pose=param_dict['right_hand_pose'].reshape(1, 45),
                transl=param_dict['transl'],
                expression=param_dict['expression'],
                jaw_pose=param_dict['jaw_pose'],
                leye_pose=param_dict['leye_pose'],
                reye_pose=param_dict['reye_pose'],
                **kwargs)
            verts = mesh_result['vertices'][0].cpu().numpy()[::4, :]
        if img is not None:
            canvas = cv2.resize(
                src=img,
                dsize=(self.fisheye_param.width, self.fisheye_param.height))
        else:
            canvas = np.zeros(
                shape=(self.fisheye_param.height, self.fisheye_param.width, 3),
                dtype=np.uint8)
        verts2d = self.projector.project(points=verts)[0]
        point_radius = 1
        for point_idx in range(verts2d.shape[0]):
            point_loc = np.around(
                verts2d[point_idx, :], decimals=0).astype(np.int32)
            cv2.circle(
                img=canvas,
                center=point_loc,
                radius=point_radius,
                color=[
                    255,
                ] * 3,
                thickness=-1)
        img = canvas
        if self.window_size is not None:
            if self.new_size is None:
                resize_ratio = \
                    self.window_size / max(img.shape[:2])
                self.new_size = (int(img.shape[1] * resize_ratio),
                                 int(img.shape[0] * resize_ratio))
            img = cv2.resize(img, self.new_size)
            cv2.imshow('SMPLVisualization', img)
            cv2.waitKey(1)
        if self.tmp_img_path is not None:
            cv2.imwrite(self.tmp_img_path, img)
        return img

    def __del__(self):
        cv2.destroyAllWindows()
