import task1_camera_calibration
import task2_3_projective_transformation
import cv2
import numpy as np

if __name__ == '__main__':
    camera_calibrator = task1_camera_calibration.CameraCalibration(
        (22, 16),
        (1440, 960),
        0.03, # in meters
        0.022)
    camera_calibrator.set_directory("./calibration/")
    undistorted = camera_calibrator.undistort_all("./images/")
    transformer = task2_3_projective_transformation.Transformer("./images/")
    matrix_0_1 = np.array([[ 8.48183861e-04, -1.19290455e-03,  2.06367310e-02],
                             [-3.00056842e-04,  3.27034491e-05, -8.66962076e-06],
                             [ 1.50623467e-02, -1.41866938e-02,  1.00000000e+00]])
    transformer.check_projective_transformation("set_1_1.jpg", matrix_0_1)
