import task1_camera_calibration

if __name__ == '__main__':
    camera_calibrator = task1_camera_calibration.CameraCalibration((16, 22), (1440, 960), 30)
    camera_calibrator.set_directory("./calibration/")
    camera_calibrator.run_detection_check()
