import unittest
import numpy as np
from binary_feature_selection import feature_selection

class TestMutualInformation(unittest.TestCase):
    def test_two_features(self):
        n = 100
        useful = np.random.randint(0, 2, n)
        useless = np.random.randint(0, 2, n)
        y = useful
        X = np.array([useful, useless]).T
        fs = feature_selection.BinaryClassFeatureSelector(method='mutual_information', max_features=1)
        fs.fit(X, y)
        X_transformed = fs.transform(X, y)
        self.assertEqual(list(useful), list(X_transformed[:,0]))
        self.assertEqual(X_transformed.shape, (n, 1))