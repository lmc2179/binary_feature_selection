import numpy as np
from binary_feature_selection._probability_summary import BinaryClassProbabilitySummary

class AbstractFeatureImportanceMetric(object):
    def fit(self, T, C):
        self.prob_summary = BinaryClassProbabilitySummary(T, C)

    def score_feature(self, i):
        raise NotImplementedError

class MutualInformation(AbstractFeatureImportanceMetric):
    def score_feature(self, i):
        t_c_joint = self.prob_summary.get_joint_probability_feature_class(i)
        t = self.prob_summary.get_prob_feature(i)
        c = self.prob_summary.get_prob_class()
        return np.log(t_c_joint / (t*c))