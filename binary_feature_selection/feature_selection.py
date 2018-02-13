from binary_feature_selection import metrics

class BinaryClassFeatureSelector(object):
    def __init__(self, method='mutual_information', max_features=10):
        self.metric = self._init_metric(method)
        self.max_features = max_features
        self.most_important_features = None

    def _init_metric(self, method):
        metric_name_mapping = {'mutual_information': metrics.MutualInformation}
        if method in metric_name_mapping:
            return metric_name_mapping[method]()
        else:
            raise Exception

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

    def fit_transform(self, X, y):
        pass