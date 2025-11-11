import task1_camera_calibration

if __name__ == '__main__':
    camera_calibrator = task1_camera_calibration.CameraCalibration(
        (16, 22),
        (1440, 960),
        0.03, # in meters
        0.0225)
    camera_calibrator.set_directory("./calibration/")
    camera_calibrator.run_detection_check_corners()
    #camera_calibrator.run_undistortion_check("0.jpg")
