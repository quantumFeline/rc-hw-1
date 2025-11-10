import cv2
import os
import numpy as np

class CameraCalibration:
    def __init__(self, board_size: tuple, image_size: tuple, checker_size: int):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.board_size = board_size
        self.image_size = image_size
        self.pattern_size = (board_size[0]-1, board_size[1]-1)
        self.checker_size = checker_size
        self.directory = ""

    def set_directory(self, directory: str):
        self.directory = directory

    def generate_object_points(self):
        points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[:self.pattern_size[0], :self.pattern_size[1]].T.reshape(-1, 2)
        return points

    def run_detection_check(self):
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            print(filename)
            image = cv2.imread(os.path.join(self.directory, filename))
            cv2.imshow(filename, image)
            cv2.waitKey(0)

            corners, ids, _ = self.detector.detectMarkers(image)
            print(len(corners), ids.shape)
            detected = cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow(filename, detected)
            cv2.waitKey(0)

    def detect_corners(self):
        all_corners = []
        all_ids = []
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            image = cv2.imread(self.directory + filename)
            corners, ids, _ = self.detector.detectMarkers(image)
            all_corners.append(corners)
            all_ids.append(ids)
        return all_corners, all_ids

    def calibrate(self):
        corners, ids = self.detect_corners()
        n = len(corners)
        object_points = self.generate_object_points()

        #print(len(corners))
        #print(corners[0])
        _, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
                objectPoints=np.array([object_points] * n),
                imagePoints=corners,
                imageSize=self.image_size,
                cameraMatrix=None,
                distCoeffs=None)
        return camera_matrix, distortion_coefficients

    def find_rectify_maps_and_roi(self):
        camera_matrix, distortion_coefficients = self.calibrate()
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, self.image_size, 0)
        print(new_camera_matrix)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix,
                                                 distortion_coefficients,
                                                 np.eye(3),
                                                 new_camera_matrix,
                                                 size=self.image_size,
                                                 m1type=cv2.CV_32FC1)
        return map1, map2, roi

    def undistort_image(self, image):
        map_from, map_to, roi = self.find_rectify_maps_and_roi()
        x, y, w, h = roi
        print("roi:", roi)
        image_undistorted = image.copy()
        image_undistorted = cv2.remap(image_undistorted, map_from, map_to, cv2.INTER_LINEAR)
        image_cropped = image_undistorted[y:y + h, x:x + w]
        return image_cropped

    def run_undistortion_check(self, image_filename):
        image_to_undistort = cv2.imread(os.path.join(self.directory, image_filename))
        image = self.undistort_image(image_to_undistort)
        cv2.imshow("undistorted", image)
        cv2.waitKey(0)