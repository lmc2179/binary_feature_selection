import numpy as np

class BinaryClassProbabilitySummary(object):
    def __init__(self, T, C):
        # TODO: Reshape?
        self.prob_t = np.mean(T, axis=0)
        self.prob_c = np.mean(C, axis=0)
        self.prob_t_c = np.mean(T[C[:,0]==1], axis=0) * self.prob_c

    def get_prob_feature(self, i):
        return self.prob_t[i]

    def get_prob_class(self):
        return self.prob_c

    def get_joint_probability_feature_class(self, i):
        return self.prob_t_c[i]