import cv2
import numpy as np
import os
import tqdm

class Transformer:

    def __init__(self, directory):
        self.directory = directory

    @staticmethod
    def find_homography(corners_src, corners_dst):
        """
        Calculates a homography matrix between two sets of corners.
        :param corners_src: corners to project from
        :param corners_dst: corners to project to
        :return: a 3x3 homography matrix
        """
        assert len(corners_src) == len(corners_dst), "Should match"
        eq_n = len(corners_src)

        A = np.zeros((eq_n * 2, 9)) # the matrix for the system of linear equations to solve.
        # Each two lines represent a transformation from a corner to a corner.
        for i in range(eq_n):
            #print(corners_src[i], corners_dst[i])
            x, y = corners_src[i]
            u, v = corners_dst[i]
            A[2*i] = np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A[2*i+1] = np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        # find the least squares solution
        s, sigma, vt = np.linalg.svd(A)
        h = vt[-1, :] # smallest eigenvector of A.T @ A

        assert h.shape == (9,), "Something went wrong with the matrix shapes. This shouldn't happen."
        matrix = h.reshape((3,3))
        return matrix / matrix[2][2] # normalize

    @staticmethod
    def apply_projective_transformation_svd(image, corners_src, corners_dst):
        """
        Calculates projective transformation of an image given a matrix.
        :param image: original image
        :param corners_src: a tuple of the corners of the source points of the image
        :param corners_dst: a tuple of the corners of the destination points of the image
        :return: a transformed matrix
        """
        matrix = Transformer.find_homography(corners_src, corners_dst)
        return Transformer.apply_projective_transformation(image, matrix)


    @staticmethod
    def apply_projective_transformation(image, matrix, output_shape = None):
        """
        Calculates projective transformation of an image given a matrix.
        :param image: original image
        :param matrix: homography matrix
        :param output_shape: output shape of the image
        :return: a transformed image
        """
        matrix_inverse = np.linalg.inv(matrix)
        matrix_inverse /= matrix_inverse[2][2]
        print("matrix:", matrix)
        print("inverse:", matrix_inverse)

        print("shape:", image.shape)
        if output_shape is None:
            destination = np.zeros(shape=image.shape)
        else:
            destination = np.zeros(shape=output_shape)

        for row in tqdm.tqdm(range(image.shape[0])):
            for col in range(image.shape[1]):
                source_coords = matrix_inverse @ np.array([col, row, 1])
                source_coords /= source_coords[2]
                source_coords = np.round(source_coords).astype(np.int32)
                source_col, source_row = source_coords[0], source_coords[1]
                if 0 <= source_row < image.shape[0] and 0 <= source_col < image.shape[1]:
                    destination[row, col, :] = image[source_row, source_col, :]
                # else leave black

        return destination.astype(np.uint8)

    @staticmethod
    def projective_transformation_opencv(image, matrix):
        """
        Calculates projective transformation of an image given a matrix.
        This version uses the standard OpenCV tool. It is only her for comparing the results.
        :param image: original image
        :param matrix: a 3x3 matrix describing the transformation
        :return: a transformed image
        """
        return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def check_projective_transformation(self, image_file, matrix) -> None:
        """
        Given an image and a matrix, checks how the projective transform works
        by displaying the original and the transformed image next to each other.
        :param image_file: original image filepath (relative to directory)
        :param matrix: transformation matrix
        :return: None
        """
        original_image = cv2.imread(os.path.join(self.directory, image_file))
        transformed_image = self.apply_projective_transformation(original_image, matrix)
        transformed_image_control = self.projective_transformation_opencv(original_image, matrix)

        print(f"Original shape: {original_image.shape}")
        print(f"Custom transform shape: {transformed_image.shape}")
        print(f"OpenCV transform shape: {transformed_image_control.shape}")

        together = np.hstack((original_image, transformed_image, transformed_image_control)).astype(np.uint8)
        print(together.shape)
        together_small = cv2.resize(together, (0, 0), fx=0.3, fy=0.3)
        cv2.imshow("Transformation", together_small)
        cv2.waitKey(0)