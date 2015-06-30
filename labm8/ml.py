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

from random import randint

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import system


MODULE_SUPPORTED = system.which("weka") and not lab.is_python3()


if MODULE_SUPPORTED:
    import weka
    import weka.core
    import weka.core.jvm as jvm
    import weka.classifiers
    from weka.classifiers import Classifier as WekaClassifier
    from weka.core.dataset import Instances as WekaInstances
    from weka.core.converters import Loader as WekaLoader
    from weka.core.converters import Saver as WekaSaver
    from weka.core.classes import Random as WekaRandom
else:
    # Add dummy variables as required.
    WekaClassifier = object
    WekaInstances = object


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
    return loader.load_file(fs.path(src))


def load_csv(src, **kwargs):
    return load(src, loader="weka.core.converters.CSVLoader", **kwargs)


def save(data, dst, saver="weka.core.converters.ArffSaver", **kwargs):
    if not MODULE_SUPPORTED: return
    saver = WekaSaver(classname=saver, **kwargs)
    saver.save_file(data, fs.path(dst))


def save_csv(src, **kwargs):
    return save(src, saver="weka.core.converters.CSVSaver", **kwargs)


def load_and_save(src, dst, loader, saver, loader_args={}, saver_args={}):
    if not MODULE_SUPPORTED: return
    data = load(src, loader=loader, **loader_args)
    save(data, dst, saver=saver, **saver_args)


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


class Dataset(object):
    """
    Dataset object, wrapper around weka.core.dataset.Instances.
    """

    def __init__(self, instances, class_index=-1):
        self.instances = instances
        self.class_index = class_index

    def __repr__(self):
        return self.instances.__repr__()

    def __self__(self):
        return self.instances.__str__()

    def __len__(self):
        return self.instances.num_instances

    def __getitem__(self, index):
        if index < 0:
            index = self.instances.num_instances + index
        return self.instances.get_instance(index)

    @property
    def class_index(self):
        return self.instances.class_index

    @class_index.setter
    def class_index(self, index):
        # If index is < 0, add value to num_attributes.
        if index < 0:
            index = self.instances.num_attributes + index
        self.instances.class_index = index

    def folds(self, nfolds=10, seed=None):
        """
        Get (training,testing) datasets for cross-validation.

        Arguments:

            nfolds (int, optional): Number of folds. Default value is
              10.
            seed (int, optional): Seed value for shuffling
              dataset. Default value is random int 0 <= x <= 10000.

        Returns:

            list of (Instances,Instances) tuples: Each list element is
              a pair of (training,testing) datasets, respectively.
        """
        seed = seed or randint(0, 10000)
        rnd = WekaRandom(seed)

        fold_size = labmath.ceil(self.instances.num_instances / nfolds)

        # Shuffle the dataset.
        instances = WekaInstances.copy_instances(self.instances)
        instances.randomize(rnd)

        folds = []
        for i in range(nfolds):
            offset = i * fold_size
            testing_end = min(offset + fold_size, instances.num_instances - 1)

            # Calculate dataset indices for testing and training data.
            testing_range = (offset, testing_end - offset)
            left_range = (0, offset)
            right_range = (testing_end, instances.num_instances - testing_end)

            # If there's nothing to test, move on.
            if testing_range[1] < 1: continue

            # Create testing and training folds.
            testing = WekaInstances.copy_instances(instances, *testing_range)
            left = WekaInstances.copy_instances(instances, *left_range)
            right = WekaInstances.copy_instances(instances, *right_range)
            training = WekaInstances.append_instances(left, right)

            # Add fold to collection.
            folds.append((training, testing))

        return folds

    def save(dst, saver="weka.core.converters.ArffSaver", **kwargs):
        save(self.instances, dst, saver=saver, **kwargs)

    @staticmethod
    def load(src, loader="weka.core.converters.ArffLoader", **kwargs):
        instances = load(src, loader=loader, **kwargs)
        return Dataset(instances)

    @staticmethod
    def load_csv(src, **kwargs):
        return Dataset.load(src, loader="weka.core.converters.CSVLoader",
                            **kwargs)


class Classifier(WekaClassifier):
    """
    Object representing a classifier.
    """

    def classify(self, instance):
        """
        Classify an instance and return the value.
        """
        value_index = self.classify_instance(instance)
        dataset = instance.dataset
        return dataset.attribute(dataset.class_index).value(value_index)

    def train(self, dataset):
        self.build_classifier(dataset.instances)

    def __repr__(self):
        return " ".join([self.classname,] + self.options)

    def __str__(self):
        return self.__repr__()


class J48(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.trees.J48"
        super(J48, self).__init__(classname=classname, *args, **kwargs)


class NaiveBayes(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.bayes.NaiveBayes"
        super(NaiveBayes, self).__init__(classname=classname, *args, **kwargs)


class ZeroR(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.rules.ZeroR"
        super(ZeroR, self).__init__(classname=classname, *args, **kwargs)


class SMO(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.functions.SMO"
        super(SMO, self).__init__(classname=classname, *args, **kwargs)


class SimpleLogistic(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.functions.SimpleLogistic"
        super(SimpleLogistic, self).__init__(classname=classname,
                                             *args, **kwargs)


class RandomForest(Classifier):
    def __init__(self, *args, **kwargs):
        classname = "weka.classifiers.trees.RandomForest"
        super(RandomForest, self).__init__(classname=classname, *args, **kwargs)
