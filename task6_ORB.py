import numpy as np
import cv2
from matplotlib import pyplot as plt
import task5_image_stitching as t5s

ORB_MAX_FEATURES = 5000
RANSAC_THRESHOLD = 5.0
LOWE_RATIO_DEFAULT = 0.7
LOWE_RATIO_RELAXED = 0.8

class ORB:
    """
    ORB-based feature detection and image stitching.
    Uses RANSAC for robust homography estimation.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
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

    def get_knn_matches(self, image1, image2, k=2):
        kp1, descr1 = self.get_keypoints_descriptions(image1)
        kp2, descr2 = self.get_keypoints_descriptions(image2)
        matches_knn = self.matcher.knnMatch(descr1, descr2, k=k)
        return kp1, kp2, matches_knn

    def show_matches(self, image1, image2):
        kp1, kp2, matches = self.get_matches(image1, image2)
        print([(match.imgIdx, match.distance, match.queryIdx, match.trainIdx) for match in matches])

        match_result = cv2.drawMatches(img1, kp1,
                                     img2, kp2, matches[:20], None, matchesThickness=5)
        match_result = cv2.resize(match_result, (1000, 650))
        cv2.imshow("matches", match_result)
        cv2.waitKey(0)

    def show_matches_knn(self, image1, image2):
        kp1, kp2, matches_knn = self.get_knn_matches(image1, image2)
        print([(match[0].imgIdx, match[0].distance, match[0].queryIdx, match[0].trainIdx) for match in matches_knn])

        match_result = cv2.drawMatchesKnn(img1, kp1,
                                     img2, kp2, matches_knn[:20], None)
        cv2.imshow("knn matches", match_result)
        cv2.waitKey(0)

    def match_with_ransac(self, image1, image2):
        """
        Use built-in ORB + RANSAC to find matches and display them
        :param image1: first image
        :param image2: second image
        :return: good matches
        """
        kp1, kp2, matches = self.get_knn_matches(image1, image2, k=2)
        # store all the good matches as per Lowe's ratio test.
        print(matches)
        good = [m for m, n in matches if m.distance < LOWE_RATIO_DEFAULT * n.distance]
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
        matchesMask = mask.ravel().tolist()
        h,w = image1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        draw_params = dict(
            matchColor=(0, 255, 0),  # green for inliers
            singlePointColor=(255, 0, 0),  # red for outliers
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        result = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
        cv2.imshow("RANSAC: green=inliers, red=outliers", cv2.resize(result, (0, 0), fx=0.6, fy=0.6))
        cv2.waitKey(0)

        return src_pts, dst_pts, good

    def stitch(self, image1, image2):
        kp1, kp2, matches = self.get_knn_matches(image1, image2, k=2)
        good = [m for m, n in matches if m.distance < LOWE_RATIO_RELAXED * n.distance]
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESHOLD)
        print(homography)
        #make 3d if 2d
        image1 = image1[:, :, None] if image1.ndim == 2 else image1
        image2 = image2[:, :, None] if image2.ndim == 2 else image2
        stitched = t5s.ImageStitcher.stitch(image1, image2, homography)
        cv2.imshow("ORB: stitched", cv2.resize(stitched, (0, 0), fx=0.3, fy=0.3))
        cv2.waitKey(0)

if __name__ == "__main__":
    img1 = cv2.imread("output/set_1_1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("output/set_1_2.jpg", cv2.IMREAD_GRAYSCALE)
    orb = ORB()
    orb.match_with_ransac(img1, img2)
    orb.stitch(img1, img2)