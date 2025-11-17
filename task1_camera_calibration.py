import cv2
import os
import numpy as np

class CameraCalibration:
    def __init__(self, board_size: tuple, image_size: tuple, checker_size: float, marker_size: float):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.marker_detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.board_size = board_size
        self.image_size = image_size
        self.pattern_size = (board_size[0]-1, board_size[1]-1)
        self.checker_size = checker_size
        self.marker_size = marker_size
        self.directory = ""

        self.charuco_board = cv2.aruco.CharucoBoard(self.board_size, self.checker_size,
                                               self.marker_size, self.dictionary)
        self.charuco_board.setLegacyPattern(True)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, detectorParams=self.parameters)
        # (TRY THIS) Increase refinement window size (Default: 5)
        # A larger window can help find the sub-pixel corner on large,
        # blurry, or distorted markers.
        self.parameters.cornerRefinementWinSize = 10

        # (TRY THIS) Increase max iterations (Default: 30)
        self.parameters.cornerRefinementMaxIterations = 50

        # (TRY THIS) Allow smaller markers (Default: 0.03)
        # The fisheye lens makes markers at the edge very small.
        # This lowers the minimum size (as a % of image dimension)
        self.parameters.minMarkerPerimeterRate = 0.01

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
            image = cv2.imread(os.path.join(self.directory, filename))
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Add this

            marker_corners, marker_ids, _ = self.marker_detector.detectMarkers(gray)

            if marker_ids is not None:
                print(f"{filename}: {len(marker_ids)} markers detected")
                print(f"  Detected IDs: {sorted(marker_ids.flatten().tolist())[:20]}...")  # First 20 IDs

                # Check what the board expects
                expected_ids = list(self.charuco_board.getIds())
                print(f"  Expected IDs: {sorted(expected_ids)[:20]}...")
                print(f"  IDs match: {set(marker_ids.flatten()).issubset(set(expected_ids))}")

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(
                gray,  # Use gray instead of image
                markerCorners=marker_corners,
                markerIds=marker_ids
            )

            if marker_ids is not None:
                print(f"{filename}: {len(marker_ids)} markers", end="")

            if charuco_ids is not None and len(charuco_ids) > 0:
                print(f", {len(charuco_ids)} corners")
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
            else:
                print(", NO corners")

        return all_corners, all_ids

        print("Preliminary calibration done. Using this as a hint.")
        print(f"Prelim Matrix: {prelim_cam_matrix}")
        print(f"Prelim Distortion: {prelim_dist_coeffs}")

        # --- PASS 2: Detect ChArUco corners using the hint ---
        print("\n--- Starting Pass 2: Guided ChArUco detection ---")
        all_charuco_corners = []
        all_charuco_ids = []

        for i, gray in enumerate(all_img_gray):
            # We already have marker corners from Pass 1
            marker_corners = all_marker_corners[i] if i < len(all_marker_corners) else []
            marker_ids = all_marker_ids[i] if i < len(all_marker_ids) else []

            if marker_ids is not None and len(marker_ids) > 0:
                # Now, call detectBoard *with the hints*
                charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
                    gray,
                    markerCorners=marker_corners,
                    markerIds=marker_ids,
                    cameraMatrix=prelim_cam_matrix,  # <-- THE HINT
                    distCoeffs=prelim_dist_coeffs  # <-- THE HINT
                )

                if charuco_ids is not None and len(charuco_ids) > 0:
                    print(f"Image {i}: Successfully interpolated {len(charuco_ids)} ChArUco corners")
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                else:
                    print(f"Image {i}: FAILED to interpolate ChArUco corners, even with hint.")
            else:
                print(f"Image {i}: No markers found (skipping).")

        if not all_charuco_corners:
            print("ERROR: Failed to detect any ChArUco corners. Calibration failed.")
            return None, None

        # --- PASS 2: Run final, high-quality ChArUco calibration ---
        print("\n--- Running Pass 2: Final ChArUco calibration ---")

        _, camera_matrix, distortion_coefficients, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=self.charuco_board,
            imageSize=self.image_size,
            cameraMatrix=None,  # Start fresh for the final calibration
            distCoeffs=None
        )

        print(f"Camera matrix: {camera_matrix}")
        print(f"Distortion: {distortion_coefficients}")
        return camera_matrix, distortion_coefficients

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

    def find_rectify_maps_and_roi(self):
        camera_matrix, distortion_coefficients = self.calibrate() # !
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
        """
        Runs a camera calibration based on the images in the directory, then undistorts a single image
        and displays the result.
        Used for debugging purposes.
        :param image_filename: filename of the image to undistort
        :return: None
        """
        image_to_undistort = cv2.imread(os.path.join(self.directory, image_filename))
        cv2.imshow("original", image_to_undistort)
        cv2.waitKey(0)
        image = self.undistort_image(image_to_undistort)
        cv2.imshow("undistorted", image)
        cv2.waitKey(0)