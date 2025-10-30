import task1_camera_calibration

if __name__ == '__main__':
    camera_calibrator = task1_camera_calibration.CameraCalibration()
    camera_calibrator.run_detection_check("./calibration/")
