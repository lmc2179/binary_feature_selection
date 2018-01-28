import unittest
import numpy as np
from binary_feature_selection import _probability_summary

class ProbabilitySummaryTest(unittest.TestCase):
    def test_prob_summary(self):
        T = np.array([[1, 0],
                     [1, 1]])
        C = np.array([[1],
                      [0]])
        summary = _probability_summary.BinaryClassProbabilitySummary(T, C)
        self.assertEqual(summary.get_prob_feature(0), 1)
        self.assertEqual(summary.get_prob_feature(1), 0.5)
        self.assertEqual(summary.get_joint_probability_feature_class(0), 0.5)
        self.assertEqual(summary.get_joint_probability_feature_class(1), 0)
        self.assertEqual(summary.get_prob_class(), 0.5)