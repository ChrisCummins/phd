#
# cgo13 - Implementation of the autotuner from:
#
#     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
#     Mapping of Data Parallel Programs to OpenCL for Heterogeneous
#     Systems. In CGO. IEEE.
#
import os

import labm8
from labm8 import ml

import smith


class BadInputException(smith.SmithException): pass


def from_arff(arff_path):
    pass


def from_csv(csv_path):
    base, extension = os.path.splitext(csv_path)
    if extension != '.csv':
        raise BadInputException("fatal: is file '{}' really a csv?"
                                .format(csv_path))

    arff_path = base + '.arff'
    ml.csv2arff(csv_path, arff_path)
    return from_arff(arff_path)
