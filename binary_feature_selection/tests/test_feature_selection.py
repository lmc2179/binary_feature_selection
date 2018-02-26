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

class TestMutualInformationSelector(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('mutual_information')

class TestCETSelector(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('cross_entropy_for_text')

class TestInformationGainSelector(AbstractFeatureSelectorTest):
    def test_two_features(self):
        self._assert_two_features('information_gain')

class AbstractInteractionTest(unittest.TestCase):
    def _assert_finds_interaction(self, method):
        n = 100
        x1 = np.random.randint(0, 2, n)
        x2 = np.random.randint(0, 2, n)
        useless = np.random.randint(0, 2, n)
        y = np.array(x1 * x2)
        X = np.array([x1, x2, useless]).T
        fs = feature_selection.PairwiseInteractionSelector(method=method, max_features=1)
        X_new = fs.fit_transform(X, y)
        self.assertEqual(list(X_new[:, -1]), list(y))

class TestMutualInformationInteraction(AbstractInteractionTest):
    def test_find_interaction(self):
        self._assert_finds_interaction('mutual_information')

class TestCETInteraction(AbstractInteractionTest):
    def test_find_interaction(self):
        self._assert_finds_interaction('cross_entropy_for_text')

class TestInformationGainInteraction(AbstractInteractionTest):
    def test_find_interaction(self):
        self._assert_finds_interaction('information_gain')