#
# cgo13 - Implementation of the autotuner from:
#
#     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
#     Mapping of Data Parallel Programs to OpenCL for Heterogeneous
#     Systems. In CGO. IEEE.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import numpy as np
import os
import re
import sys

from io import StringIO
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import labm8
from labm8 import math as labmath

import smith


class BadInputException(smith.SmithException): pass


# Feature extractors:

def cgo13_features(d, with_raw_features=False):
    return [
        float(d["F1:transfer/(comp+mem)"]),
        float(d["F2:coalesced/mem"]),
        float(d["F3:(localmem/mem)*avgws"]),
        float(d["F4:comp/mem"])
    ]


def cgo13_with_raw_features(d, with_raw_features=False):
    return [
        int(d["comp"]),
        int(d["rational"]),
        int(d["mem"]),
        int(d["localmem"]),
        int(d["coalesced"]),
        int(d["atomic"]),
        int(d["transfer"]),
        int(d["wgsize"]),
        float(d["F1:transfer/(comp+mem)"]),
        float(d["F2:coalesced/mem"]),
        float(d["F3:(localmem/mem)*avgws"]),
        float(d["F4:comp/mem"])
    ]


def raw_features(d, with_raw_features=False):
    return [
        int(d["comp"]),
        int(d["rational"]),
        int(d["mem"]),
        int(d["localmem"]),
        int(d["coalesced"]),
        int(d["atomic"]),
        int(d["transfer"]),
        int(d["wgsize"])
    ]


def labels(d):
    return d["oracle"]


class Metrics(object):
    def __init__(self, prefix, data, predicted):
        self._prefix = prefix
        self._data = data
        self._predicted = predicted

    @property
    def prefix(self): return self._prefix

    @property
    def data(self): return self._data

    @property
    def predicted(self): return self._predicted

    @property
    def oracles(self):
        try:
            return self._oracles
        except AttributeError:
            self._oracles = np.array([float(d["speedup"]) for d in self.data])
            return self._oracles

    @property
    def oracle(self):
        try:
            return self._oracle
        except AttributeError:
            self._oracle = self.speedup / labmath.geomean(self.oracles)
            return self._oracle

    @property
    def y_test(self):
        try:
            return self._y_test
        except AttributeError:
            self._y_test = np.array([d["oracle"] for d in self.data])
            return self._y_test

    @property
    def accuracy(self):
        try:
            return self._accuracy
        except AttributeError:
            self._accuracy = accuracy_score(self.y_test, self.predicted)
            return self._accuracy

    @property
    def speedups(self):
        try:
            return self._speedups
        except AttributeError:
            speedups = []
            for d,p in zip(self.data, self.predicted):
                if d["oracle"] == p:
                    speedups.append(float(d["speedup"]))
                else:
                    speedups.append(float(d["penalty"]))
            self._speedups = np.array(speedups)
            return self._speedups

    @property
    def speedup(self):
        try:
            return self._speedup
        except AttributeError:
            self._speedup = labmath.geomean(self.speedups)
            return self._speedup

    header = ", ".join([
        "classifier",
        "accuracy",
        "speedup",
        "oracle"
    ])

    def __repr__(self):
        return ", ".join([
            self.prefix,
            "{:.2f}%".format(self.accuracy * 100),
            "{:.2f}".format(self.speedup),
            "{:.0f}%".format(self.oracle * 100)
        ])


def getsuite(d):
    return re.match(r"^[a-zA-Z-]+-[0-9\.]+", d["benchmark"]).group(0)


def pairwise(iterable):
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def getgroups(data, getgroup):
    return sorted(list(set([getgroup(d) for d in data])))


def group_xval_folds(data, getgroup):
    groups = getgroups(data, getgroup)

    g = defaultdict(list)
    for i,d in enumerate(data):
        g[getgroup(d)].append(i)
    g = sorted(list(g.values()), key=lambda x: getgroup(data[x[0]]))

    pairs = []
    for j in range(len(g)):
        for i in range(len(g)):
            # Note that we're not excluding cases where j == i, so
            # train and test data will be identical for # groups of
            # the folds.
            pairs.append((g[j], g[i]))
    return pairs


def run_fold(prefix, clf, data, train_index, test_index,
             features=cgo13_features):
    X_train = np.array([features(data[i]) for i in train_index])
    y_train = np.array([labels(data[i]) for i in train_index])

    clf.fit(X_train, y_train)
    X_test = np.array([features(data[i]) for i in test_index])
    y_test = np.array([labels(data[i]) for i in test_index])

    predicted = clf.predict(X_test)
    predicted_data = [d for i,d in enumerate(data) if i in test_index]

    return Metrics(prefix, predicted_data, predicted)


def run_test(prefix, clf, train, test, features=cgo13_features):
    X_train = np.array([features(d) for d in train])
    y_train = np.array([labels(d) for d in train])

    clf.fit(X_train, y_train)
    X_test = np.array([features(d) for d in test])
    y_test = np.array([labels(d) for d in test])

    predicted = clf.predict(X_test)

    return Metrics(prefix, test, predicted)


def run_xval(prefix, clf, data, cv, features=cgo13_features, seed=1):
    X = np.array([features(d) for d in data])
    y = np.array([labels(d) for d in data])

    predicted = cross_validation.cross_val_predict(clf, X, y, cv=cv)

    return Metrics("DecisionTree", data, predicted)


class ZeroR(object):
    # TODO:
    pass


def classification(train, test=None, with_raw_features=False,
                   group_by=None, zeror=False, **kwargs):
    if with_raw_features:
        features = cgo13_with_raw_features
    else:
        features = cgo13_features

    seed = kwargs.get("seed", 0)

    X_train = np.array([features(d, with_raw_features=with_raw_features)
                        for d in train])
    y_train = np.array([labels(d) for d in train])

    if zeror:
        clf = ZeroR()
    else:
        clf = DecisionTreeClassifier(
            criterion="entropy", splitter="best", random_state=seed)

    if test:
        print(Metrics.header)
        print(run_test("DecisionTree", clf, train, test,
                       features=features))
    elif group_by:
        # Cross-validation over some grouping
        getgroup = {
            "suite": getsuite
        }.get(group_by, None)
        if group_by and not getgroup:
            raise(smith.SmithException("Unkown group type '{}'"
                                       .format(group_by)))

        folds = group_xval_folds(train, getgroup)
        groups = getgroups(train, getgroup)
        results = np.zeros((len(groups), len(groups)))

        for train_index, test_index in folds:
            train_group = getgroup(train[train_index[0]])
            test_group = getgroup(train[test_index[0]])

            metrics = run_fold("DecisionTree", clf, train,
                               train_index, test_index,
                               features=features)
            results[groups.index(train_group), groups.index(test_group)] = metrics.oracle

        print("-", ",".join(groups), sep=",")
        for i,row in enumerate(results):
            print(groups[i], ",".join(["{:.2f}%".format(x * 100) for x in row]), sep=",")

    else:
        # Plain old cross-validation.
        print(Metrics.header)
        folds = cross_validation.KFold(len(train), n_folds=10,
                                       random_state=seed)
        print(run_xval("DecisionTree", clf, train, folds,
                       features=features))


def from_csv(csv_path):
    return smith.read_csv(smith.assert_exists(csv_path))
