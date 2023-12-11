import cv2
import numpy as np
import torch
from xrprimer.data_structure.camera import PinholeCameraParameter
from xrprimer.utils.log_utils import get_logger

from .camera import Pinhole2D
from .rasterizer import estimate_normals, has_rasterizer, import_exception
from .utils import vis_normals


class MinimalVerticesRenderer:

    def __init__(self,
                 faces,
                 device=None,
                 pinhole_param: PinholeCameraParameter = None,
                 logger=None):
        self.logger = get_logger(logger)
        if not has_rasterizer:
            self.logger.error(import_exception)
            raise ImportError
        if pinhole_param is None:
            self.logger.warning('No pinhole_param argument given,' +
                                'use default camera.')
            pinhole_param = PinholeCameraParameter(name='default_cam', )
            pinhole_param.set_intrinsic(
                width=1280, height=720, fx=5000, fy=5000, cx=112, cy=112)
        self.pinhole_param = pinhole_param
        self.pinhole2d = Pinhole2D(
            K=np.array(pinhole_param.get_intrinsic(3)),
            h=pinhole_param.height,
            w=pinhole_param.width)
        self.device = torch.device(device)
        if isinstance(faces, torch.Tensor):
            self.faces = faces.clone().detach().to(
                self.device, dtype=torch.int32)
        else:
            self.faces = torch.tensor(
                faces, dtype=torch.int32, device=self.device)
        if not self.pinhole_param.world2cam:
            self.pinhole_param.inverse_extrinsic()
        self.r_tensor = torch.tensor(
            self.pinhole_param.get_extrinsic_r(),
            dtype=torch.float32,
            device=self.device)
        self.t_tensor = torch.tensor(
            self.pinhole_param.get_extrinsic_t(),
            dtype=torch.float32,
            device=self.device)

    def __call__(self, vertices, bg=None, **kwargs):
        assert vertices.device == self.faces.device
        vertices = vertices.clone()
        vert_rot = self.r_tensor
        vert_trans = self.t_tensor.unsqueeze(0)
        vertices = vertices @ vert_rot.transpose(0, 1) + vert_trans
        coords, normals = estimate_normals(
            vertices=vertices, faces=self.faces, pinhole=self.pinhole2d)
        vis = vis_normals(coords, normals)
        if bg is not None:
            mask = coords[:, :, [2]] <= 0
            vis = (
                vis[:, :, None] +
                torch.tensor(bg).to(mask.device) * mask).cpu().numpy().astype(
                    np.uint8)
        else:
            # convert gray to 3 channel img
            vis = vis.detach().cpu().numpy()
            vis = cv2.merge((vis, vis, vis))
        return vis
