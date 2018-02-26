import numpy as np
from binary_feature_selection._probability_summary import BinaryClassProbabilitySummary

class IFeatureImportanceMetric(object):
    def fit(self, T, C):
        raise NotImplementedError

    def score_feature(self, i):
        raise NotImplementedError

class MutualInformation(IFeatureImportanceMetric):
    def fit(self, T, C):
        self.prob_summary = BinaryClassProbabilitySummary(T, C)

    def score_feature(self, i):
        t_c_joint = self.prob_summary.get_joint_probability_feature_class(i)
        t = self.prob_summary.get_prob_feature(i)
        c = self.prob_summary.get_prob_class()
        return np.log(t_c_joint / (t*c))

class CrossEntropyText(IFeatureImportanceMetric):
    def fit(self, T, C):
        self.prob_summary = BinaryClassProbabilitySummary(T, C)

    def score_feature(self, i):
        t_c_joint = self.prob_summary.get_joint_probability_feature_class(i)
        t = self.prob_summary.get_prob_feature(i)
        c = self.prob_summary.get_prob_class()
        if t_c_joint == 0:
            return 0
        return t_c_joint * np.log(t_c_joint / (t*c))

class InformationGain(IFeatureImportanceMetric):
    def fit(self, T, C):
        self.cet = CrossEntropyText()
        self.inv_cet = CrossEntropyText()
        self.cet.fit(T, C)
        self.inv_cet.fit(1-T, C)

    def score_feature(self, i):
        return self.cet.score_feature(i) + self.inv_cet.score_feature(i)

class ChiSquared(IFeatureImportanceMetric):
    def fit(self, T, C):
        self.prob_summaries = [[BinaryClassProbabilitySummary(T, C), BinaryClassProbabilitySummary(T, 1-C)],
                               [BinaryClassProbabilitySummary(1-T, C), BinaryClassProbabilitySummary(1-T, 1-C)]]

    def score_feature(self, i):
        num = (self.prob_summaries[0][0].get_joint_probability_feature_class(i)*self.prob_summaries[1][1].get_joint_probability_feature_class(i) + self.prob_summaries[0][1].get_joint_probability_feature_class(i)*self.prob_summaries[1][0].get_joint_probability_feature_class(i))**2
        denom = self.prob_summaries[0][0].get_joint_probability_feature_class(i)*self.prob_summaries[1][1].get_joint_probability_feature_class(i) * self.prob_summaries[0][1].get_joint_probability_feature_class(i)*self.prob_summaries[1][0].get_joint_probability_feature_class(i)
        return num / denom

class GSS(IFeatureImportanceMetric):
    def fit(self, T, C):
        self.prob_summaries = [[BinaryClassProbabilitySummary(T, C), BinaryClassProbabilitySummary(T, 1-C)],
                               [BinaryClassProbabilitySummary(1-T, C), BinaryClassProbabilitySummary(1-T, 1-C)]]

    def score_feature(self, i):
        return self.prob_summaries[0][0].get_joint_probability_feature_class(i)*self.prob_summaries[1][1].get_joint_probability_feature_class(i) - self.prob_summaries[0][1].get_joint_probability_feature_class(i)*self.prob_summaries[1][0].get_joint_probability_feature_class(i)