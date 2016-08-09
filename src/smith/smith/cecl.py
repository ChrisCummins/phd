#
# libcecl utilities.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import labm8
from labm8 import fs

import smith


class InterpretException(smith.SmithException): pass


def log2features(log, out=sys.stdout, metaout=sys.stderr):
    return {
        "benchmark": "",
        "dataset": "",
        "comp": "",
        "rational": "",
        "mem": "",
        "localmem": "",
        "coalesced": "",
        "atomic": "",
        "transfer": "",
        "wgsize": "",
        "F1:transfer/(comp+mem)": "",
        "F2:coalesced/mem": "",
        "F3:(localmem/mem)*avgws": "",
        "F4:comp/mem": "",
        "runtime": "",
        "run": ""
    }


def dir2features(log, out=sys.stdout, metaout=sys.stderr):
    pass
