import cv2
import numpy as np
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.camera.distortion import undistort_camera


def get_undistort_maps(fisheye_param: FisheyeCameraParameter):
    pinhole_param = undistort_camera(fisheye_param)
    dist_coeff_np = np.array(fisheye_param.get_dist_coeff())
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=np.array(fisheye_param.get_intrinsic(3)),
        distCoeffs=dist_coeff_np,
        R=np.eye(3),
        newCameraMatrix=np.array(pinhole_param.get_intrinsic(3)),
        size=np.array((
            pinhole_param.width,
            pinhole_param.height,
        )),
        m1type=cv2.CV_32FC1)
    return map1, map2
