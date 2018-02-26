import heapq
import numpy as np
from binary_feature_selection import metrics

class AbstractMetricFeatureSelector(object):
    def __init__(self, method='mutual_information', max_features=10):
        self.metric = self._init_metric(method)
        self.max_features = max_features
        self.most_important_features = None

    def _init_metric(self, method):
        metric_name_mapping = {'mutual_information': metrics.MutualInformation,
                               'cross_entropy_for_text': metrics.CrossEntropyText,
                               'information_gain': metrics.InformationGain,
                               'chi2': metrics.ChiSquared,
                               'gss': metrics.GSS}
        if method in metric_name_mapping:
            return metric_name_mapping[method]()
        else:
            raise Exception

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X, y):
        raise NotImplementedError

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

class BinaryClassFeatureSelector(AbstractMetricFeatureSelector):
    def fit(self, X, y):
        self.metric.fit(X, y)
        _, c = X.shape
        scores_and_features = []
        for i in range(c):
            score = self.metric.score_feature(i)
            scores_and_features.append((score, i))
        self.most_important_features = [index for _, index in sorted(scores_and_features, reverse=True)][:self.max_features]

    def transform(self, X, y):
        return X[:, self.most_important_features]

class PairwiseInteractionSelector(AbstractMetricFeatureSelector):
    def fit(self, X, y): #TODO: Break up
        _, c = X.shape
        scored_interactions = []
        for i1 in range(c):
            for i2 in range(i1+1, c):
                t = X[:,i1] * X[:,i2]
                self.metric.fit(t.reshape(-1, 1), y)
                score = self.metric.score_feature(0)
                if len(scored_interactions) == self.max_features:
                    heapq.heappushpop(scored_interactions, (score, i1, i2))
                else:
                    heapq.heappush(scored_interactions, (score, i1, i2))
        self.most_important_features = [(i1, i2) for _, i1, i2 in scored_interactions]

    def transform(self, X, y):
        interactions = np.zeros((len(X), len(self.most_important_features)))
        for j, tup in enumerate(self.most_important_features):
            i1, i2 = tup
            interactions[:,j] = X[:,i1] * X[:,i2]
        return np.concatenate((X, interactions), axis=1)