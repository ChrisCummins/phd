# Copyright (C) 2015, 2016 Chris Cummins.
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
from unittest import main
from tests import TestCase

import sys

import labm8 as lab
from labm8 import fs
from labm8 import io

from labm8 import ml


class TestML(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestML, self).__init__(*args, **kwargs)

        # Make a copy of test data.
        fs.cp("tests/data/diabetes.arff", "/tmp/labm8.diabetes.arff")
        fs.cp("tests/data/diabetes.csv", "/tmp/labm8.diabetes.csv")

        # Test data paths.
        self.arff = "/tmp/labm8.diabetes.arff"
        self.csv = "/tmp/labm8.diabetes.csv"

    @classmethod
    def setUpClass(cls):
        # Calling start() multiple times should not be a problem.
        ml.start()
        ml.start()

    @classmethod
    def tearDownClass(cls):
        # Calling stop() multiple times should not be a problem.
        ml.stop()
        ml.stop()

    # csv2arff()
    def test_csv2arff(self):
        if not ml.MODULE_SUPPORTED: return
        ml.csv2arff(self.csv, "/tmp/labm8.arfftest")
        a = open("tests/data/diabetes.csv2arff")
        b = open("/tmp/labm8.arfftest")
        self._test(a.read(), b.read())
        a.close()
        b.close()

    # arff2csv()
    def test_arff2csv(self):
        if not ml.MODULE_SUPPORTED: return
        ml.arff2csv(self.arff, "/tmp/labm8.csvtest")
        a = open(self.csv)
        b = open("/tmp/labm8.csvtest")
        self._test(a.read(), b.read())
        a.close()
        b.close()

    # classifier_basename()
    def test_classifier_basename(self):
        self._test("ZeroR", ml.classifier_basename("weka.classifiers."
                                                   "rules.ZeroR"))
        self._test("SMO", ml.classifier_basename("weka.classifiers."
                                                 "functions.SMO -C 1.0"))

    # Classifier
    def test_classifier(self):
        if not ml.MODULE_SUPPORTED: return
        j48 = ml.Classifier("weka.classifiers.trees.J48")

        # str() representation
        self._test("weka.classifiers.trees.J48 -C 0.25 -M 2", str(j48))

        # basename property
        self._test("J48", j48.basename)

        # Train on dataset
        dataset = ml.Dataset.load(self.arff)
        dataset.class_index = -1
        j48.train(dataset)

        # Test on a couple of instances.
        self._test("tested_positive", j48.classify(dataset[0]))
        self._test("tested_negative", j48.classify(dataset[-1]))

    # Dataset
    def test_dataset_len(self):
        if not ml.MODULE_SUPPORTED: return
        dataset = ml.Dataset.load(self.arff)
        self._test(768, len(dataset))

    def test_dataset_folds(self):
        if not ml.MODULE_SUPPORTED: return
        dataset = ml.Dataset.load(self.arff)
        folds = dataset.folds(nfolds=10)

        self._test(10, len(folds))
        for training,testing in folds:
            io.debug("Training:", training.num_instances,
                     "Testing:", testing.num_instances)
            self._test(True, training.num_instances > testing.num_instances)


if __name__ == '__main__':
    main()
