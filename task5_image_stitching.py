import cv2
import task2_3_projective_transformation as t2p
import numpy as np

# POINTS_1 = [(340, 638), (743, 633), (261, 695), (270, 793), (169, 800), (165, 696), (335, 421), (337, 615), (157, 623), (152, 428)]
# POINTS_2 = [(399, 788), (795, 797), (322, 847), (333, 937), (240, 941), (230, 850), (392, 585), (395, 765), (225, 763), (220, 581)]
POINTS_1 = [(246, 522), (401, 520), (408, 691), (254, 692), (262, 803), (328, 804), (334, 874), (266, 875), (420, 728), (841, 740), (264, 217), (437, 174)]
POINTS_2 = [(551, 566), (695, 577), (691, 747), (548, 730), (550, 838), (610, 847), (611, 918), (550, 907), (701, 785), (1156, 861), (585, 280), (752, 240)]

class ImageStitcher:

    def __init__(self, image_filename_1: str, points_1: list[tuple[int, int]], image_filename_2: str, points_2: list[tuple[int,int]]) -> None:
        self.im1 = cv2.imread(image_filename_1)
        self.im2 = cv2.imread(image_filename_2)
        self.points_1 = points_1
        self.points_2 = points_2
        self.matrix = t2p.Transformer.find_homography(self.points_2, self.points_1)

    @staticmethod
    def get_corners(image):
        print("image shape:", image.shape)
        return [[0, 0], [0, image.shape[0]], [image.shape[1], 0], [image.shape[1], image.shape[0]]]

    def blend(self,
              canvas_size: tuple[int, int, int],
              im1_offset: tuple[int, int],
              im1: cv2.Mat,
              im2: cv2.Mat) -> cv2.Mat:
        """
        Blends two images together. When both images have a pixel at a given point, the algorithm takes the
        arithmetic average
        :param canvas_size: size of the output image
        :param im1_offset: offset of the first image relative to the second image
        :param im1: first image to blend
        :param im2: second image to blend
        :return: blended image
        """
        print("canvas_size:", canvas_size, "im1 size:", im1.shape, "im2 size:", im2.shape)
        im1_shifted = np.zeros(canvas_size, dtype=np.uint8)

        # Lay on the first image
        h1, w1 = im1.shape[:2]
        offset_y = im1_offset[0]
        offset_x = im1_offset[1]
        im1_shifted[offset_y:offset_y + h1, offset_x:offset_x + w1] = self.im1

        mask1 = (im1_shifted.sum(axis=2) > 0).astype(np.float32)
        mask2 = (im2.sum(axis=2) > 0).astype(np.float32)
        overlap = mask1 * mask2
        stitched_image = np.where(
            overlap[:, :, np.newaxis] > 0,
            (im1_shifted * 0.5 + im2 * 0.5).astype(np.uint8), # if both present, take average
            np.where(mask2[:, :, np.newaxis] > 0, im2, # if im2 present, take it
                     im1_shifted) # otherwise, take im1
        )
        return stitched_image

    def stitch(self):
        im1_corners = self.get_corners(self.im1)
        im2_projected_corners = [[v[0]/v[2], v[1]/v[2]] for v in [(self.matrix @ np.array([corner[0], corner[1], 1])) for corner in self.get_corners(self.im2)]]
        corners = np.array(im1_corners + im2_projected_corners)
        print(corners)
        print("corners[:,0]:", corners[:,0], "corners[:,1]:", corners[:,1])
        min_x = np.floor(np.min(corners[:, 0])).astype(np.int32)
        max_x =  np.ceil(np.max(corners[:, 0])).astype(np.int32)
        min_y = np.floor(np.min(corners[:, 1])).astype(np.int32)
        max_y =  np.ceil(np.max(corners[:, 1])).astype(np.int32)
        print("min_x:", min_x, "min_y:", min_y, "max_x:", max_x, "max_y:", max_y)
        translation = np.array([[1, 0, -min_x],
                                [0, 1, -min_y],
                                [0, 0, 1]])

        matrix_with_translation = translation @ self.matrix # homography first!
        canvas_size = (max_y - min_y, max_x - min_x, 3)

        im2_projected = t2p.Transformer.apply_projective_transformation(self.im2, matrix_with_translation, output_shape=canvas_size)

        result = self.blend(canvas_size,
                            (0 - min_y, 0 - min_x),
                            self.im1,
                            im2_projected)

        cv2.imshow("result", result)
        cv2.waitKey(0)

stitcher = ImageStitcher("output/set_3_1.jpg", POINTS_1, "output/set_3_2.jpg", POINTS_2)
stitcher.stitch()