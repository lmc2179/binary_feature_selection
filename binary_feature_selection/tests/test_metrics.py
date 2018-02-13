import unittest
import numpy as np
import random
from binary_feature_selection import metrics

class TestMutualInformation(unittest.TestCase):
    def test_two_features(self):
        n = 100
        useful = np.random.randint(0, 2, n)
        useless = np.random.randint(0, 2, n)
        y = useful
        X = np.array([useful, useless]).T
        mi = metrics.MutualInformation()
        mi.fit(X, y)
        self.assertGreater(mi.score_feature(0), mi.score_feature(1))