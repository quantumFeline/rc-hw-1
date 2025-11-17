import unittest
import numpy as np

def random_quadrilateral(width, height):
    return np.stack((np.random.randint(0, width, size=(4,)),
                           np.random.randint(0, height, size=(4,)))).astype(np.uint32)

@unittest
def test_projectiveTransformation