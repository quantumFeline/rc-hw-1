import numpy as np
import cv2
from matplotlib import pyplot as plt

class ORB:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher()

    def get_keypoints_descriptions(self, image):
        kp = self.orb.detect(image, None)
        return self.orb.compute(image, kp)

    def show_keypoints(self, image1, image2):
        kp1 = self.orb.detect(image1, None)
        kp2 = self.orb.detect(image2, None)

        img1_points = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=0)
        img2_points = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=0)
        f, arr = plt.subplots(1, 2)
        arr[0].imshow(img1_points)
        arr[1].imshow(img2_points)
        plt.show()

    def get_matches(self, image1, image2):
        kp1, descr1 = self.get_keypoints_descriptions(image1)
        kp2, descr2 = self.get_keypoints_descriptions(image2)

        matches = self.matcher.match(descr1, descr2)
        return kp1, kp2, matches

    def get_knn_matches(self, image1, image2):
        kp1, descr1 = self.get_keypoints_descriptions(image1)
        kp2, descr2 = self.get_keypoints_descriptions(image2)
        matches_knn = self.matcher.knnMatch(descr1, descr2, k=2)
        return kp1, kp2, matches_knn

    def show_matches(self, image1, image2):
        kp1, kp2, matches = self.get_matches(image1, image2)
        print([(match.imgIdx, match.distance, match.queryIdx, match.trainIdx) for match in matches])

        match_result = cv2.drawMatches(img1, kp1,
                                     img2, kp2, matches[:20], None, matchesThickness=5)
        match_result = cv2.resize(match_result, (1000, 650))
        cv2.imshow("result", match_result)
        cv2.waitKey(0)

    def show_matches_knn(self, image1, image2):
        kp1, kp2, matches_knn = self.get_knn_matches(image1, image2)
        print([(match[0].imgIdx, match[0].distance, match[0].queryIdx, match[0].trainIdx) for match in matches_knn])

        match_result = cv2.drawMatchesKnn(img1, kp1,
                                     img2, kp2, matches_knn[:20], None)
        cv2.imshow("result", match_result)
        cv2.waitKey(0)

img1 = cv2.imread("output/set_1_1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("output/set_1_2.jpg", cv2.IMREAD_GRAYSCALE)
orb = ORB()
orb.show_matches(img1, img2)