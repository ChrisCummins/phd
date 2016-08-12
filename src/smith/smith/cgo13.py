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

import sys
import os

from io import StringIO

from weka.filters import Filter
from weka.classifiers import Classifier,FilteredClassifier,PredictionOutput
from weka.classifiers import Evaluation
from weka.core.classes import Random

import labm8
from labm8 import ml
from labm8 import math as labmath

import smith


class BadInputException(smith.SmithException): pass


IGNORED_ATTRIBUTES = [
    "benchmark",
    "dataset",
    "runtime",
    "speedup",
    "penalty"
]

UNUSED_ATTRIBUTES = IGNORED_ATTRIBUTES + [
    "comp",
    "rational",
    "mem",
    "localmem",
    "coalesced",
    "atomic",
    "transfer",
    "wgsize"
]


def eval_classifier(base_classifier, arff, test_arff=None,
                    ignored_attributes=UNUSED_ATTRIBUTES,
                    seed=1):
    classifier = FilteredClassifier()

    ignored_indices = [arff.attribute_by_name(x).index
                       for x in ignored_attributes]
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                    options=["-R", ','.join(
                        [str(x + 1) for x in ignored_indices])])
    classifier.filter = remove
    classifier.classifier = base_classifier

    # Additional prediction output. Print the speedup and penalty for
    # all instances.
    extra_attributes = [
        "runtime",
        "speedup",
        "penalty"
    ]
    extra_indices = [arff.attribute_by_name(x).index for x in extra_attributes]
    pout = PredictionOutput(
        classname="weka.classifiers.evaluation.output.prediction.PlainText",
        options=["-p", ','.join([str(x + 1) for x in extra_indices])])

    evl = Evaluation(arff)

    if test_arff:
        print("Training on {} instances, testing on {} "
              "({:.1f}:1.0 train/test ratio)"
              .format(arff.num_instances,
                      test_arff.num_instances,
                      arff.num_instances / test_arff.num_instances),
              file=sys.stderr)
        classifier.build_classifier(arff)
        evl.test_model(classifier, test_arff, pout)
    else:
        # Number of splits in training data. For leave-one-out, use
        # arff.num_instances:
        nfolds = 10
        print("Running {}-fold validation on {} instances"
              .format(nfolds, arff.num_instances), file=sys.stderr)
        evl.crossvalidate_model(classifier, arff, nfolds, Random(seed), pout)


    assert(len(evl.predictions), arff.num_instances)

    basename = ml.classifier_basename(base_classifier.classname)
    num_attributes = arff.num_attributes - len(ignored_attributes)

    def parse_line(x):
        """
        Parse a line of output from Weka PredictionOutput.
        """
        c = x.split()
        actual = c[1]
        predicted = c[2]

        if len(c) == 6:
            c.remove('+')

        runtime, speedup, penalty = c[4].split(',')
        runtime = float(runtime[1:])
        speedup = float(speedup)
        penalty = float(penalty[:-1])

        return {
            "actual": actual,
            "predicted": predicted,
            "runtime": runtime,
            "penalty": penalty,
            "oracle": speedup,
            "speedup": speedup if actual == predicted else penalty
        }

    predictions = [parse_line(x) for x in str(pout).split('\n')[1:-1]]

    speedups = [x["speedup"] for x in predictions]
    oracles = [x["oracle"] for x in predictions]
    runtimes = [x["runtime"] for x in predictions]

    rmean = labmath.mean(runtimes)

    # Speedups of predicted.
    pmean = labmath.mean(speedups)
    pgeo = labmath.geomean(speedups)
    pmin = min(speedups)
    pmax = max(speedups)
    ci = labmath.confinterval(speedups, array_mean=pmean)[1] - pmean

    # Oracle speedups.
    omean = labmath.mean(oracles)
    ogeo = labmath.geomean(oracles)
    omin = min(oracles)
    omax = max(oracles)

    print(basename,
          "{:.2f}".format(evl.percent_correct),
          "{:.2f}".format(rmean),
          "{:.2f}".format(pgeo),
          "{:.2f}".format(pmean),
          "{:.2f}".format(pmin),
          "{:.2f}".format(pmax),
          "{:.2f}".format(ci),
          "{:.2f}".format(pgeo / ogeo),
          "{:.2f}".format(ogeo),
          "{:.2f}".format(omean),
          "{:.2f}".format(omin),
          "{:.2f}".format(omax),
          sep=",")


def classification(arff, with_raw_features=False, **kwargs):
    classifiers = [
        Classifier(classname="weka.classifiers.rules.ZeroR"),
        Classifier(classname="weka.classifiers.trees.J48",
                   options=["-C", "0.5"]),
        Classifier(classname="weka.classifiers.lazy.IBk",
                   options=["-K", "3",
                            "-W", "0",
                            "-A", "weka.core.neighboursearch.LinearNNSearch"])
    ]

    # Print header.
    print(
        "classifier",
        "accuracy",
        "avgruntime",
        "geospeedup",
        "avgspeedup",
        "minspeedup",
        "maxspeedup",
        "ci",
        "oracle",
        "geooracle",
        "avgoracle",
        "minoracle",
        "maxoracle",
        sep=",")

    ignored_attributes = (IGNORED_ATTRIBUTES if with_raw_features else
                          UNUSED_ATTRIBUTES)

    kwargs["ignored_attributes"] = ignored_attributes
    for classifier in classifiers:
        eval_classifier(classifier, arff, **kwargs)


def from_arff(arff_path):
    data = ml.load(arff_path)
    data.class_index = data.attribute_by_name("oracle").index
    return data


def from_csv(csv_path):
    base, extension = os.path.splitext(csv_path)
    if extension != '.csv':
        raise BadInputException("is file '{}' really a csv?".format(csv_path))

    arff_path = base + '.arff'
    ml.csv2arff(csv_path, arff_path)
    return from_arff(arff_path)
