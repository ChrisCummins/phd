#
# libcecl utilities.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import labm8
from labm8 import fs

import smith


class InterpretException(smith.SmithException): pass


def log2features(log, metaout=sys.stderr):
    return {
        "benchmark": None,
        "dataset": None,
        "comp": None,
        "rational": None,
        "mem": None,
        "localmem": None,
        "coalesced": None,
        "atomic": None,
        "transfer": None,
        "wgsize": None,
        "F1:transfer/(comp+mem)": None,
        "F2:coalesced/mem": None,
        "F3:(localmem/mem)*avgws": None,
        "F4:comp/mem": None,
        "runtime": None,
        "run": None
    }
