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

from glob import glob, iglob
from labm8 import fs

import clgen
from clgen import config as cfg
from clgen import dbutil
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen import torch_rnn
from clgen.cache import Cache
from clgen.model import Model


def serialize_argspec(args):
    names = map(chr, range(97, 97 + len(args)))
    strings = [arg + " " + name for arg, name in zip(args, names)]
    return "__kernel void A({args}) {{".format(args=", ".join(strings))


class Sampler(clgen.CLgenObject):
    def __init__(self, sampler_opts, kernel_opts):
        assert(type(sampler_opts) is dict)
        assert(type(kernel_opts) is dict)

        checksum_data = sorted(
            [str(x) for x in list(sampler_opts.values())] +
            [str(x) for x in list(kernel_opts.values())])
        string = "".join([str(x) for x in checksum_data])
        self.hash = clgen.checksum_str(string)

        # parse sampler options
        self.max_kernels = sampler_opts.get("max_kernels", -1)
        self.batch_size = sampler_opts.get("batch_size", 1000)
        self.static_checker = sampler_opts.get("static_checker", True)
        self.dynamic_checker = sampler_opts.get("dynamic_checker", False)

        if self.dynamic_checker and not cfg.USE_OPENCL:
            log.warning("dynamic checking requested, but OpenCL not available")
            self.dynamic_checker = False

        self.kernel_opts = kernel_opts

    def get_cache(self, model):
        return Cache(fs.path(model.hash, model.corpus.hash, self.hash))

    def sample_iteration(self, model):
        assert(isinstance(model, Model))

        cache = self.get_cache(model)

        # create samples database if it doesn't exist
        if not cache["kernels.db"]:
            dbutil.create_db(fs.path(cache.path, "kernels.tmp.db"))
            cache["kernels.db"] = fs.path(
                cache.path, "kernels.tmp.db")

        if self.kernel_opts.get("args", None) is not None:
            start_text = serialize_argspec(self.kernel_opts["args"])
        else:
            start_text = "__kernel void A("

        # sample options
        opts = {
            "opencl": 1,
            "stream": 1,
            "n": self.batch_size,
            "checkpoint": model.most_recent_checkpoint,
            "temperature": self.kernel_opts.get("temperature", .75),
            "length": self.kernel_opts.get("max_length", 10000),
            "start_text": start_text,
        }

        tmppath = fs.path(cache.path, "sample.tmp.cl")
        torch_rnn.sample(tmppath, **opts)
        fetch.process_sample_file(cache["kernels.db"], tmppath,
                                  max_kernel_len=opts["length"])

        # TODO: Parse static checker requirement
        preprocess.preprocess_db(cache["kernels.db"])
        fs.rm(tmppath)

    def sample(self, model):
        cache = self.get_cache(model)

        # create samples database if it doesn't exist
        if not cache["kernels.db"]:
            dbutil.create_db(fs.path(cache.path, "kernels.tmp.db"))
            cache["kernels.db"] = fs.path(
                cache.path, "kernels.tmp.db")

        while True:
            if (self.max_kernels > 0 and
                dbutil.num_good_kernels(cache["kernels.db"]) > self.max_kernels):
                return

            self.sample_iteration(model)

        log.info("samples database:", cache["kernels.db"])


def from_json(sampler_json):
    sampler_opts = sampler_json.get("sampler", None)
    if not sampler_opts:
        raise clgen.UserError("no sampler section in sampler specification")

    kernel_opts = sampler_json.get("kernels", None)
    if not kernel_opts:
        raise clgen.UserError("no kernels section in sampler specification")

    return Sampler(sampler_opts, kernel_opts)
