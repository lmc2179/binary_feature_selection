import unittest
import numpy as np
from binary_feature_selection import feature_selection

class AbstractFeatureSelectorTest(unittest.TestCase):
    def _assert_two_features(self, method_name):
        n = 100
        useful = np.random.randint(0, 2, n)
        useless = np.random.randint(0, 2, n)
        y = useful
        X = np.array([useful, useless]).T
        fs = feature_selection.BinaryClassFeatureSelector(method=method_name, max_features=1)
        fs.fit(X, y)
        X_transformed = fs.transform(X, y)
        self.assertEqual(list(useful), list(X_transformed[:,0]))
        self.assertEqual(X_transformed.shape, (n, 1))

class TestMutualInformation(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('mutual_information')

class TestCET(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('cross_entropy_for_text')

class TestInformationGain(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('information_gain')
