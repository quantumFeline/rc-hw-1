import unittest
import numpy as np
from task2_3_projective_transformation import Transformer
import tqdm

class TransformationTest(unittest.TestCase):

    @staticmethod
    def random_points(width, height, n):
        return np.stack((np.random.rand(n) * width, np.random.rand(n) * height), axis=0).astype(np.float32).T
        # values from 0 to width and from 0 to height

    @staticmethod
    def random_matrix():
        matrix = np.random.rand(3, 3)
        return matrix / matrix[2][2]

    @staticmethod
    def calc_transform(source_points, matrix):
        destination_points = []
        for point in source_points:
            v = np.array((point[0], point[1], 1))
            res = np.dot(matrix, v)
            destination_points.append((res[0] / res[2], res[1] / res[2]))
        return np.array(destination_points)

    def test_projective_transformation(self):
        source_points = TransformationTest.random_points(10, 10, 100)
        matrix = TransformationTest.random_matrix()
        destination_points = TransformationTest.calc_transform(source_points, matrix)

        found_matrix = Transformer.find_homography(source_points, destination_points)

        # Both matrices should be already normalized.
        assert np.allclose(matrix, found_matrix)

    def test_run_100_test_check(self):
        for i in tqdm.tqdm(range(100)):
            self.test_projective_transformation()

