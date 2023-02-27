"""
Decision trees

Here we shall research decision trees

Held by: Yevhenii Bevz / Khrystyna Mysak
"""

import pandas as pd
import numpy as np
from collections import Counter


class Node:

    def __init__(self, X, y, gini, num_samples, num_samples_per_class, predicted_class):
        self.X = X
        self.y = y
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class MyDecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth


    def gini(self, groups, classes):
        """
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.

        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        """

        pass


    def split_data(self, X, y) -> tuple[int, int]:
        # test all the possible splits in O(N*F) where N in number of samples
        # and F is number of features

        # return index and threshold value

        feature_size = y.size
        if feature_size <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.object_classes)]

        the_best_gini = 1 - sum((n / feature_size)**2 for n in num_parent)
        best_index, best_threshold = None, None

        for index in range(self.object_features):
            thresholds, classes = zip(*sorted(zip(X[:, index], y)))

            on_the_left = [0] * self.object_classes
            on_the_right = num_parent.copy()

            for i in range(1, feature_size):
                c = classes[i - 1]
                on_the_left[c] += 1
                on_the_right[c] -= 1
                left_gini = 1 - sum((on_the_left[x] / i)**2 for x in range(self.object_classes))
                right_gini = 1 - sum((on_the_right[x] / i)**2 for x in range(self.object_classes))

                gini = (i*left_gini + (feature_size - i)*right_gini) / feature_size

                if thresholds[i] == thresholds[i-1]:
                    continue

                if gini < the_best_gini:
                    the_best_gini = gini
                    best_index = index
                    best_threshold = (thresholds[i] + thresholds[i - 2]) / 2

    def build_tree(self, X, y, depth=0):
        # create a root node

        # recursively split until max depth is not exeeced

        pass


    def fit(self, X, y):
        # basically wrapper for build tree / train

        pass


    def predict(self, X_test):
        # traverse the tree while there is a child
        # and return the predicted class for it,
        # note that X_test can be a single sample or a batch

        pass


    def evaluate(self, X_test, y_test):
        # return accuracy

        pass