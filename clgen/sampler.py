#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
import sys

from glob import glob, iglob
from labm8 import fs
from labm8 import system

import clgen
from clgen import config as cfg
from clgen import dbutil
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen.cache import Cache
from clgen.explore import explore
from clgen.model import Model


def serialize_argspec(args: list) -> str:
    """
    Serializes an argument spec to a kernel prototype.

    Arguments:
        args (str[]): Argument specification.

    Returns:
        str: Kernel prototype.
    """
    names = map(chr, range(97, 97 + len(args)))
    strings = [arg + " " + name for arg, name in zip(args, names)]
    return "__kernel void A({args}) {{".format(args=", ".join(strings))


class Sampler(clgen.CLgenObject):
    """
    CLgen sampler for models.
    """
    def __init__(self, sampler_opts: dict, kernel_opts: dict):
        """
        Instantiate a sampler.

        Arguments:
            sampler_opts (dict): Sampler options.
            kernel_opts (dict): Kernel options.
        """
        assert(type(sampler_opts) is dict)
        assert(type(kernel_opts) is dict)

        self.hash = self._hash(sampler_opts, kernel_opts)

        # parse sampler options
        self.max_kernels = sampler_opts.get("max_kernels", -1)
        self.batch_size = sampler_opts.get("batch_size", 1000)
        self.max_batches = sampler_opts.get("max_batches", -1)
        self.static_checker = sampler_opts.get("static_checker", True)
        self.dynamic_checker = sampler_opts.get("dynamic_checker", False)

        if self.dynamic_checker and not cfg.USE_OPENCL:
            log.warning("dynamic checking requested, but OpenCL not available")
            self.dynamic_checker = False

        self.kernel_opts = kernel_opts

    def _hash(self, sampler_opts: dict, kernel_opts: dict) -> str:
        """compute sampler checksum"""
        checksum_data = sorted(
            [str(x) for x in sampler_opts.values()] +
            [str(x) for x in kernel_opts.values()])
        string = "".join([str(x) for x in checksum_data])
        return clgen.checksum_str(string)

    def cache(self, model: Model) -> Cache:
        """
        Return sampler cache.

        Arguments:
            model (Model): CLgen model.

        Returns:
            Cache: Cache.
        """
        sampler_model_hash = clgen.checksum_str(self.hash + model.hash)
        return Cache(fs.path("sampler", sampler_model_hash))

    def sample_iteration(self, model: Model, quiet: bool=False) -> None:
        """
        Run one sample iteration.

        Arguments:
            model (Model): CLgen model.
        """
        assert(isinstance(model, Model))

        cache = self.cache(model)

        if self.kernel_opts.get("args", None):
            start_text = serialize_argspec(self.kernel_opts["args"])
        else:
            start_text = "__kernel void A("

        tmppath = fs.path(cache.path,
                          "sampler-{pid}.tmp.cl".format(pid=system.PID))

        with open(tmppath, "w") as outfile:
            opts = {
                "output": outfile,
                "num_samples": self.batch_size,
                "temperature": self.kernel_opts.get("temperature", 1),
                "max_length": self.kernel_opts.get("max_length", 10000),
                "seed_text": start_text,
                "quiet": quiet
            }
            model.sample(**opts)

        sys.stdout.flush()
        sys.stderr.flush()
        fetch.process_sample_file(cache["kernels.db"], tmppath,
                                  max_kernel_len=opts["max_length"], quiet=True)

        if self.static_checker:
            # TODO: Parse dynamic checker requirement
            preprocess.preprocess_db(cache["kernels.db"])
        fs.rm(tmppath)

    def sample(self, model: Model, quiet: bool=False) -> None:
        """
        Sample CLgen model.

        Arguments:
            model (Model): CLgen model.
        """
        cache = self.cache(model)

        # create samples database if it doesn't exist
        if not cache["kernels.db"]:
            dbutil.create_db(fs.path(cache.path, "kernels.tmp.db"))
            cache["kernels.db"] = fs.path(
                cache.path, "kernels.tmp.db")

        batch_i = 0
        while True:
            # stop if we have enough kernels
            has_max_kernels = self.max_kernels >= 0
            num_good_kernels = dbutil.num_good_kernels(cache["kernels.db"])
            if has_max_kernels and num_good_kernels >= self.max_kernels:
                return

            # stop if we've done enough batches
            has_max_batches = self.max_batches >= 0
            if has_max_batches and batch_i >= self.max_batches:
                return

            batch_i += 1
            print("sample batch", batch_i, "...")

            self.sample_iteration(model, quiet=quiet)

            print()
            explore(self.cache(model)["kernels.db"])

        log.info("samples database:", cache["kernels.db"])


def from_json(sampler_json: dict) -> Sampler:
    """
    Instantiate sampler from JSON.

    Arguments:
        sampler_json (dict): JSON data.

    Returns:
        Sampler: Instantiate sampler.
    """
    sampler_opts = sampler_json.get("sampler", {})

    kernel_opts = sampler_json.get("kernels", {})
    if not kernel_opts:
        raise clgen.UserError("no kernels section in sampler specification")

    return Sampler(sampler_opts, kernel_opts)
