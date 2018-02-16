import unittest
import numpy as np
import random
from binary_feature_selection import metrics

class AbstractMetricTest(unittest.TestCase):
    def _assert_correct_two_features(self, metric_obj):
        n = 100
        useful = np.random.randint(0, 2, n)
        useless = np.random.randint(0, 2, n)
        y = useful
        X = np.array([useful, useless]).T
        metric_obj.fit(X, y)
        self.assertGreater(metric_obj.score_feature(0), metric_obj.score_feature(1))

class TestMutualInformation(AbstractMetricTest):
    def test_two_features(self):
        self._assert_correct_two_features(metrics.MutualInformation())

class TestCrossEntropyText(AbstractMetricTest):
    def test_two_features(self):
        self._assert_correct_two_features(metrics.CrossEntropyText())

class TestInformationGain(AbstractMetricTest):
    def test_two_features(self):
        self._assert_correct_two_features(metrics.InformationGain())