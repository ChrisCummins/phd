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

import numpy as np
import os
import pandas as pd
import re
import math
import sys

from collections import defaultdict
from io import StringIO
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import labm8
from labm8 import math as labmath

import smith


class BadInputException(smith.SmithException): pass


def ingroup(getgroup, d, group):
    return getgroup(d) == group


def getsuite(d):
    return re.match(r"^[a-zA-Z-]+-[0-9\.]+", d["benchmark"]).group(0)


def getprog(d):
    return re.match(r"^[a-zA-Z-]+-[0-9\.]+-[^-]+-", d["benchmark"]).group(0)


def getclass(d):
    return d["oracle"]


def suite_filter(d):
    """
    Groups filter.
    """
    return getsuite(d).startswith("rodinia") # or getsuite(d).startswith("parboil")


class DataFilter(object):
    @staticmethod
    def from_str(string):
        pass


class GetGroup(object):
    @staticmethod
    def from_str(string):
        pass


def _eigenvectors(data):
    F1 = data["F1_norm"]
    F2 = data["F2_norm"]
    F3 = data["F3_norm"]
    F4 = data["F4_norm"]

    # Eigenvectors: (hardcoded values as returned by Weka)
    V1 = - .1881 * F1 + .6796 * F2 - .2141 * F3 - .6760 * F4
    V2 = - .7282 * F1 - .0004 * F2 + .6852 * F3 + .0149 * F4
    V3 = - .6590 * F1 - .1867 * F2 - .6958 * F3 - .2161 * F4
    V4 =   .0063 * F1 + .7095 * F2 + .0224 * F3 - .7044 * F4

    return (V1, V2, V3, V4)


def normalize(array):
    factor = np.amax(array)
    return np.copy(array) / factor


class LabelledData(object):
    @staticmethod
    def from_csv(path, group_by=None, filter_by=None):
        datafilter = {
            "suite": suite_filter
        }.get(filter_by, None)

        getgroup = {
            "class": getclass,
            "suite": getsuite,
            "prog": getprog
        }.get(group_by, lambda x: "None")

        data = pd.read_csv(smith.assert_exists(path))

        if datafilter:
            # TODO: data filter
            pass

        # Add group column.
        data["Group"] = [getgroup(d) for d in data.to_dict(orient='records')]

        # Add normalized feature columns.
        data["F1_norm"] = normalize(data["F1:transfer/(comp+mem)"])
        data["F2_norm"] = normalize(data["F2:coalesced/mem"])
        data["F3_norm"] = normalize(data["F3:(localmem/mem)*avgws"])
        data["F4_norm"] = normalize(data["F4:comp/mem"])

        # Add eigenvectors.
        data["E1"], data["E2"], data["E3"], data["E4"] = _eigenvectors(data)

        return data


def feature_distance(f1, f2):
    """
    Distance between two features (as dicts).
    """
    d1 = abs(f1["F1_norm"] - f2["F1_norm"])
    d2 = abs(f1["F2_norm"] - f2["F2_norm"])
    d3 = abs(f1["F3_norm"] - f2["F3_norm"])
    d4 = abs(f1["F4_norm"] - f2["F4_norm"])

    return math.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4)


def nearest_neighbours(data1, data2, same_class=False,
                       distance=feature_distance):
    """
    Find the minimum distances between datapoints.

    Returns list of tuples, where each tuple is in the form:

       (distance, index_of_closest, same_oracle)
    """
    dists, indices, sameoracles = [], [], []

    for d1 in data1.to_dict(orient="record"):
        mindist, index, sameoracle = float('inf'), None, False
        for i,d2 in enumerate(data2.to_dict(orient="record")):
            if not d1 == d2:
                dist = distance(d1, d2)
                if ((not same_class) or
                    (same_class and d1["oracle"] == d2["oracle"])):
                    if dist < mindist and i not in indices:
                        mindist = dist
                        index = i
                        sameoracle = d1["oracle"] == d2["oracle"]
        dists.append(mindist)
        indices.append(index)
        sameoracles.append(sameoracle)
    return zip(dists, indices, sameoracles)


# Feature extractors:

def cgo13_features(d):
    return np.array([
        d["F1:transfer/(comp+mem)"],
        d["F2:coalesced/mem"],
        d["F3:(localmem/mem)*avgws"],
        d["F4:comp/mem"]
    ]).T


def cgo13_with_raw_features(d):
    return [
        d["comp"],
        d["rational"],
        d["mem"],
        d["localmem"],
        d["coalesced"],
        d["atomic"],
        d["transfer"],
        d["wgsize"],
        d["F1:transfer/(comp+mem)"],
        d["F2:coalesced/mem"],
        d["F3:(localmem/mem)*avgws"],
        d["F4:comp/mem"]
    ]


def raw_features(d):
    return [
        d["comp"],
        d["rational"],
        d["mem"],
        d["localmem"],
        d["coalesced"],
        d["atomic"],
        d["transfer"],
        d["wgsize"]
    ]


def getlabels(d):
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
        return self.data["speedup"]

    @property
    def oracle(self):
        try:
            return self._oracle
        except AttributeError:
            self._oracle = self.speedup / labmath.geomean(self.oracles)
            return self._oracle

    @property
    def y_test(self):
        return self.data["oracle"]

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
            for d,p in zip(self.data.to_dict(orient="records"), self.predicted):
                if d["oracle"] == p:
                    speedups.append(d["speedup"])
                else:
                    speedups.append(d["penalty"])
            self._speedups = np.array(speedups)
            return self._speedups

    @property
    def speedup(self):
        try:
            return self._speedup
        except AttributeError:
            self._speedup = labmath.geomean(self.speedups)
            return self._speedup

    @property
    def groups(self):
        try:
            return self._groups
        except AttributeError:
            self._groups = sorted(set(self.data["Group"]))
            return self._groups

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



def getgroups(data, getgroup):
    return sorted(list(set([getgroup(d) for d in data.to_dict(orient="records")])))


def pairwise_groups_indices(data, getgroup):
    """
    """
    groups = getgroups(data, getgroup)

    group_indices = defaultdict(list)
    for i,d in enumerate(data.to_dict(orient="records")):
        group_indices[getgroup(d)].append(i)

    groupnames, pairs = [], []
    for j in range(len(groups)):
        for i in range(len(groups)):
            l, r = groups[j], groups[i]
            groupnames.append((l, r))
            li, ri = group_indices[l], group_indices[r]
            pairs.append((li, ri))
    return groupnames, pairs


def run_fold(prefix, clf, data, train_index, test_index,
             features=cgo13_features):
    X_train = features(data)[train_index]
    y_train = getlabels(data)[train_index]

    clf.fit(X_train, y_train)
    X_test = features(data)[test_index]
    y_test = getlabels(data)[test_index]

    predicted = clf.predict(X_test)

    predicted_data = data.ix[test_index]

    return Metrics(prefix, predicted_data, predicted)


def run_test(prefix, clf, train, test, features=cgo13_features):
    X_train = features(train)
    y_train = getlabels(train)

    clf.fit(X_train, y_train)
    X_test = features(test)
    y_test = getlabels(test)

    predicted = clf.predict(X_test)

    return Metrics(prefix, test, predicted)


def run_xval(prefix, clf, data, cv, features=cgo13_features, seed=1):
    X = features(data)
    y = getlabels(data)

    predicted = cross_validation.cross_val_predict(clf, X, y, cv=cv)

    return Metrics("DecisionTree", data, predicted)


class ZeroR(object):
    # TODO:
    pass


def classification(train, test=None, with_raw_features=False,
                   group_by=None, zeror=False, **kwargs):
    if with_raw_features:
        getfeatures = cgo13_with_raw_features
    else:
        getfeatures = cgo13_features

    seed = kwargs.get("seed", 0)

    if zeror:
        clf = ZeroR()
    else:
        clf = DecisionTreeClassifier(
            criterion="entropy", splitter="best", random_state=seed)

    if test is not None:
        return run_test("DecisionTree", clf, train, test, features=getfeatures)
    elif group_by:
        # Cross-validation over some grouping
        getgroup = {
            "suite": getsuite
        }.get(group_by, None)
        if group_by and not getgroup:
            raise(smith.SmithException("Unkown group type '{}'"
                                       .format(group_by)))

        groupnames, folds = pairwise_groups_indices(train, getgroup)
        groups = getgroups(train, getgroup)
        results = np.zeros((len(groups), len(groups)))

        for gpname, fold in zip(groupnames, folds):
            train_group, test_group = gpname
            train_index, test_index = fold

            metrics = run_fold("DecisionTree", clf, train,
                               train_index, test_index,
                               features=getfeatures)
            results[groups.index(train_group), groups.index(test_group)] = metrics.oracle

        return results
    else:
        # Plain old cross-validation.
        folds = cross_validation.KFold(len(train), n_folds=10,
                                       random_state=seed)
        return run_xval("DecisionTree", clf, train, folds, features=getfeatures)
