#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Sample a CLgen model.
"""
from __future__ import print_function

import os
import re

from copy import copy
from glob import glob, iglob
from labm8 import fs

import clgen
from clgen import log
from clgen import torch_rnn
from clgen.cache import Cache
from clgen.model import Model


class Sampler(clgen.CLgenObject):
    def __init__(self, sampler_opts, kernel_opts):
        assert(type(sampler_opts) is dict)
        assert(type(kernel_opts) is dict)

        self.batch_size = sampler_opts.get("batch_size", 1000)
        self.static_checker = sampler_opts.get("static_checker", True)
        self.dynamic_checker = sampler_opts.get("dynamic_checker", False)


    def sample(self, model):
        assert(isinstance(model, Model))

        print("sampling")


def from_json(sampler_json):
    sampler_opts = sampler_json.get("sampler", None)
    if not sampler_opts:
        raise clgen.UserError("no sampler section in sampler specification")

    kernel_opts = sampler_json.get("kernels", None)
    if not kernel_opts:
        raise clgen.UserError("no kernels section in sampler specification")

    return Sampler(sampler_opts, kernel_opts)
