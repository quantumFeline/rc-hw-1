import task1_camera_calibration
import task2_3_projective_transformation
import cv2
import numpy as np

if __name__ == '__main__':
    # camera_calibrator = task1_camera_calibration.CameraCalibration(
    #     (22, 16),
    #     (1440, 960),
    #     0.03, # in meters
    #     0.022)
    #camera_calibrator.set_directory("./calibration/")
    #undistorted = camera_calibrator.undistort_all("./images/", "./output/")
    transformer = task2_3_projective_transformation.Transformer("./output/")
    matrix_0_1 = np.array([[-1.68303735e+00, -1.19949272e-01,  7.53467430e+02],
                         [-8.27634129e-01 , 4.39213782e-02 , 3.15562544e+02],
                         [-2.24602461e-03, -1.54001012e-04 , 1.00000000e+00]])
    transformer.check_projective_transformation("set_1_1.jpg", matrix_0_1)
