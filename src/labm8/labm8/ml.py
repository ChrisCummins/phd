# Copyright (C) 2015 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
"""
Machine learning module.
"""
from __future__ import division

import atexit

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import system

#
MODULE_SUPPORTED = system.which("weka") and not lab.is_python3()

if MODULE_SUPPORTED:
    import weka
    import weka.core
    import weka.core.jvm as jvm
    from weka.classifiers import Classifier as WekaClassifier
    from weka.core.converters import Loader as WekaLoader
    from weka.core.converters import Saver as WekaSaver


class Error(Exception):
    """
    Module-level error.
    """
    pass


def start(*args, **kwargs):
    """
    Open a weka connection.

    May be called multiple times, but not after calling stop().

    Arguments:

        *args, **kwargs: Any additional arguments to pass to
          jvm.start().
    """
    if MODULE_SUPPORTED:
        jvm.start(*args, **kwargs)


def stop():
    """
    Stop a weka connection.

    May be called multiple times, but note that a new connection
    cannot be started after calling this.
    """
    if MODULE_SUPPORTED:
        jvm.stop()


def load(src, loader="weka.core.converters.ArffLoader", **kwargs):
    if not MODULE_SUPPORTED: return
    loader = WekaLoader(classname=loader, **kwargs)
    return loader.load_file(src)


def load_csv(src, **kwargs):
    return load(src, loader="weka.core.converters.CSVLoader", **kwargs)


def save(data, dst, saver="weka.core.converters.ArffSaver", **kwargs):
    if not MODULE_SUPPORTED: return
    saver = WekaSaver(classname=saver, **kwargs)
    saver.save_file(data, dst)


def save_csv(src, **kwargs):
    return save(src, saver="weka.core.converters.CSVSaver", **kwargs)


def load_and_save(src, dst, loader, saver, loader_args={}, saver_args={}):
    if not MODULE_SUPPORTED: return
    data = load(src, loader, **loader_args)
    save(data, dst, saver, **saver_args)


def csv2arff(csv_path, arff_path):
    """
    Export a CSV file to ARFF.
    """
    load_and_save(csv_path, arff_path,
                  "weka.core.converters.CSVLoader",
                  "weka.core.converters.ArffSaver")


def arff2csv(arff_path, csv_path):
    """
    Export an ARFF file to CSV.
    """
    load_and_save(arff_path, csv_path,
                  "weka.core.converters.ArffLoader",
                  "weka.core.converters.CSVSaver")


class Classifier(object):
    """
    Object representing a classifier.
    """
    def __init__(self, classname, data, *args):
        """
        """
        self.data = data
        # Create the classifier.
        self.classifier = WekaClassifier(classname=classname, options=args)
        # Train on data.
        self.classifier.build_classifier(self.data)
        # Index of class labels.
        # TODO: Determine from dataset.
        self.label_index = self.data.num_attributes - 1

    def classify(self, instance):
        value_index = self.classifier.classify_instance(instance)
        return self.data.attribute(self.label_index).value(value_index)


class J48(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.trees.J48"
        super(J48, self).__init__(classname, *args, **kwargs)


class NaiveBayes(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.bayes.NaiveBayes"
        super(NaiveBayes, self).__init__(classname, *args, **kwargs)


def evaluate(classifier, testing):
    """
    Return a list of speedups for a J48 classifier trained on
    "training_data" and tested using "testing_data".
    """
    return [classify(classifier, instance, testing) for instance in testing]
