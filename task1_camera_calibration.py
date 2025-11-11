import cv2
import os
import numpy as np

class CameraCalibration:
    def __init__(self, board_size: tuple, image_size: tuple, checker_size: float, marker_size: float):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.board_size = board_size
        self.image_size = image_size
        self.pattern_size = (board_size[0]-1, board_size[1]-1)
        self.checker_size = checker_size
        self.marker_size = marker_size
        self.directory = ""

        self.charuco_board = cv2.aruco.CharucoBoard(self.board_size, self.checker_size,
                                               self.marker_size, self.dictionary)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)

    def set_directory(self, directory: str):
        self.directory = directory

    def generate_object_points(self):
        points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[:self.pattern_size[0], :self.pattern_size[1]].T.reshape(-1, 2)
        return points

    def run_detection_check_markers(self):
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            print(filename)
            image = cv2.imread(os.path.join(self.directory, filename))
            cv2.imshow(filename, image)
            cv2.waitKey(0)

            corners, ids, _ = self.marker_detector.detectMarkers(image)
            print(len(corners), ids.shape)
            detected = cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow(filename, detected)
            cv2.waitKey(0)

    def run_detection_check_corners(self):
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            print(f"Checking out {filename}")
            image = cv2.imread(os.path.join(self.directory, filename))
            cv2.imshow(filename, image)
            cv2.waitKey(0)

            marker_corners, marker_ids, _ = self.marker_detector.detectMarkers(image)

            corners, ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(image,
                                                                                         markerCorners=marker_corners,
                                                                                         markerIds=marker_ids)
            if ids is not None and len(ids) > 0:
                print(len(corners), ids.shape)
                detected = cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
                cv2.imshow(filename, detected)
                cv2.waitKey(0)
                detected_corners = cv2.aruco.drawDetectedCornersCharuco(corners, ids)
                cv2.imshow(filename, detected_corners)
                cv2.waitKey(0)

    def detect_markers(self):
        all_corners = []
        all_ids = []
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            image = cv2.imread(self.directory + filename)
            corners, ids, _ = self.marker_detector.detectMarkers(image)
            print(f"{filename}: detected {len(ids)} markers")
            all_corners.append(corners)
            all_ids.append(ids)
        print(f"Total images with markers: {len(all_corners)}")
        return all_corners, all_ids

    def detect_corners(self):
        all_corners = []
        all_ids = []

        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            image = cv2.imread(os.path.join(self.directory, filename))

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(image)

            if marker_ids is not None:
                print(f"{filename}: {len(marker_ids)} markers", end="")

            if charuco_ids is not None and len(charuco_ids) > 0:
                print(f", {len(charuco_ids)} corners")
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
            else:
                print(", NO corners")

        return all_corners, all_ids

    def calibrate(self):
        corners, ids = self.detect_corners()

        _, camera_matrix, distortion_coefficients, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=corners,
            charucoIds=ids,
            board=self.charuco_board,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        print(f"Camera matrix: {camera_matrix}")
        print(f"Distortion: {distortion_coefficients}")
        return camera_matrix, distortion_coefficients

    # def calibrate(self):
    #     markers, ids = self.detect_markers()
    #     n = len(markers)
    #     object_points = self.generate_object_points()
    #
    #     board = cv2.aruco.GridBoard(
    #         size=self.board_size,
    #         markerLength=self.marker_size,
    #         markerSeparation=self.checker_size - self.marker_size,  # gap between markers
    #         dictionary=self.dictionary
    #     )
    #     #print(len(corners))
    #     #print(corners[0])
    #     obj_points = []
    #     img_points = []
    #
    #     for frame_corners, frame_ids in zip(markers, ids):
    #         # Get 3D positions of detected markers from the board
    #         obj_pts, img_pts = board.matchImagePoints(frame_corners, frame_ids)
    #
    #         if obj_pts is not None and len(obj_pts) > 0:
    #             obj_points.append(obj_pts)
    #             img_points.append(img_pts)
    #
    #     _, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
    #             objectPoints=obj_points,
    #             imagePoints=img_points,
    #             imageSize=self.image_size,
    #             cameraMatrix=None,
    #             distCoeffs=None)
    #     print(f"Camera matrix: {camera_matrix}")
    #     print(f"Distortion: {distortion_coefficients}")
    #     return camera_matrix, distortion_coefficients

    def find_rectify_maps_and_roi(self):
        camera_matrix, distortion_coefficients = self.calibrate()
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, self.image_size, 0)
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
        cv2.imshow("original", image_to_undistort)
        cv2.waitKey(0)
        image = self.undistort_image(image_to_undistort)
        cv2.imshow("undistorted", image)
        cv2.waitKey(0)