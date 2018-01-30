import unittest
from random import random
import numpy as np
from binary_feature_selection import _probability_summary

class ProbabilitySummaryTest(unittest.TestCase):
    def _assert_correct_two_features(self, T, C, zero_marginal, one_marginal, zero_joint, one_joint, class_marginal):
        summary = _probability_summary.BinaryClassProbabilitySummary(T, C)
        self.assertEqual(summary.get_prob_feature(0), zero_marginal)
        self.assertEqual(summary.get_prob_feature(1), one_marginal)
        self.assertEqual(summary.get_joint_probability_feature_class(0), zero_joint)
        self.assertEqual(summary.get_joint_probability_feature_class(1), one_joint)
        self.assertEqual(summary.get_prob_class(), class_marginal)

    def test_prob_summary(self):
        T = np.array([[1, 0],
                     [1, 1]])
        C = np.array([[1],
                      [0]])
        self._assert_correct_two_features(T,
                                          C,
                                          zero_marginal=1,
                                          one_marginal=0.5,
                                          zero_joint=0.5,
                                          one_joint=0,
                                          class_marginal=0.5)

    def test_prob_summary_large_sample(self):
        gen_random_balanced = lambda n: np.array(sorted([0]*(int(n/2)) + [1]*(int(n/2)), key=lambda k: random()))
        x = gen_random_balanced(5000)
        T = np.vstack((x, 1-x)).T
        C = x.reshape(-1, 1)
        self._assert_correct_two_features(T,
                                          C,
                                          zero_marginal=0.5,
                                          one_marginal=0.5,
                                          zero_joint=0.5,
                                          one_joint=0,
                                          class_marginal=0.5)