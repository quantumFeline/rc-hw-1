import cv2
import os

class CameraCalibration:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def run_detection_check(self, directory):
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            image = cv2.imread(directory + filename)
            cv2.imshow(filename, image)
            cv2.waitKey(0)

            corners, ids, _ = self.detector.detectMarkers(image)
            print(len(corners), ids.shape)
            detected = cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow(filename, detected)
            cv2.waitKey(0)  

    def calibrate(self, directory):
        all_corners = []
        all_ids = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            image = cv2.imread(directory + filename)
            corners, ids, _ = self.detector.detectMarkers(image)
            all_corners.append(corners)
            all_ids.append(ids)
        return all_corners, all_ids