import cv2
import numpy as np
import os

class Transformer:

    def __init__(self, directory):
        self.directory = directory

    @staticmethod
    def find_homography(corners_src, corners_dst):
        """
        Calculates a homography matrix between two sets of quadrilateral corners.
        :param corners_src: corners to project from
        :param corners_dst: corners to project to
        :return: a 3x3 homography matrix
        """
        assert len(corners_src) == len(corners_dst) == 4, "Not quadrilaterals"

        A = np.zeros((8,9)) # the matrix for the system of linear equations to solve.
        # Each two lines represent a transformation from a corner to a corner.
        for i in range(4):
            x, y = corners_src[i]
            u, v = corners_dst[i]
            A[2*i] = np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A[2*i+1] = np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        # find the least squares solution
        s, sigma, vt = np.linalg.svd(A)
        h = vt[-1, :]
        return h

    @staticmethod
    def projective_transformation_svd(image, corners_src, corners_dst):
        """
        Calculates projective transformation of an image given a matrix.
        :param image: original image
        :param corners_src: a tuple of the corners of the source quadrilateral to project
        :param corners_dst: a tuple of the corners of the resulting transformation
        :return: a transformed image
        """
        corners_y = [corner[0] for corner in corners_dst]
        corners_x = [corner[1] for corner in corners_dst]
        destination_size = np.array([(max(corners_y) - min(corners_y), max(corners_x) - min(corners_x))])
        matrix_inverse = Transformer.find_homography(corners_src, corners_dst)

        destination = np.zeros(shape=destination_size)
        for dest_y in range(image.shape[0]):
            for dest_x in range(image.shape[1]):
                source_coords = matrix_inverse * (dest_x, dest_y, 1)
                source_x = source_coords[0] / source_coords[2]
                source_y = source_coords[1] / source_coords[2]
                destination[dest_y][dest_x] = image[source_y][source_x]

        return destination

    @staticmethod
    def projective_transformation(image, matrix):
        """
        Calculates projective transformation of an image given a matrix.
        :param image: original image
        :param matrix: homography matrix
        :return: a transformed image
        """
        matrix_inverse = np.linalg.inv(matrix)

        destination = np.zeros(shape=image.size)

        for dest_y in range(image.shape[0]):
            for dest_x in range(image.shape[1]):
                source_coords = matrix_inverse * (dest_x, dest_y, 1)
                source_x = source_coords[0] / source_coords[2]
                source_y = source_coords[1] / source_coords[2]
                destination[dest_y][dest_x] = image[source_y][source_x]

        return destination

    @staticmethod
    def projective_transformation_opencv(image, matrix):
        """
        Calculates projective transformation of an image given a matrix.
        This version uses the standard OpenCV tool. It is only her for comparing the results.
        :param image: original image
        :param matrix: a 3x3 matrix describing the transformation
        :return: a transformed image
        """
        return cv2.warpPerspective(image, matrix, image.size, flags=cv2.INTER_LINEAR)

    def check_projective_transformation(self, image_file, matrix) -> None:
        """
        Give an image and a matrix, checks how the projective transform works
        by displaying the original and the transformed image next to each other.
        :param image_file: original image filepath (relative to directory)
        :param matrix: transformation matrix
        :return: None
        """
        image = cv2.imread(os.path.join(self.directory, image_file))
        transformed_image = self.projective_transformation(image, matrix)
        transformed_image_control = self.projective_transformation_opencv(image, matrix)
        together = np.hstack((image, transformed_image, transformed_image_control))
        cv2.imshow("Transformation", together)
        cv2.waitKey(0)