from __future__ import division

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import math as labmath

import weka
import weka.core
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import Classifier

import omnitune
import db

from omnitune import llvm

jvm.start()
_LOADER = Loader(classname="weka.core.converters.ArffLoader")

def load_arff(path):
    """
    Return a weka-wrapper Instances object for the dataset at "path".
    """
    arff = _LOADER.load_file(path)
    arff.class_is_last()
    return arff


def create_classifier(training, weka_name, *args):
    classifier = Classifier(classname=weka_name,
                            options=args)
    classifier.build_classifier(training)
    return classifier


def classify(classifier, instance, dataset):
    label_index = dataset.num_attributes - 1
    value_index = classifier.classify_instance(instance)
    return dataset.attribute(label_index).value(value_index)


def evaluate(classifier, testing):
    """
    Return a list of speedups for a J48 classifier trained on
    "training_data" and tested using "testing_data".
    """
    return [classify(classifier, instance, testing) for instance in testing]
