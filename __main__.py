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
    matrix_0_1 = np.array([[0.8056145184130631, -0.05385100808056643, 102.88205193959797], [-0.008367707050434492, 0.6888667285519479, 239.13799248044504], [-4.915028832167057e-05, -0.00019638820521876145, 1.0]])
    transformer.check_projective_transformation("set_1_1.jpg", matrix_0_1)
