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

from copy import deepcopy
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

# Default options used for sampler. Any values provided by the user will
# override these defaults.
DEFAULT_SAMPLER_OPTS = {
    "max_kernels": -1,
    "max_batches": -1,
    "batch_size": 1000,
    "static_checker": True,
    "dynamic_checker": False,
    "gpuverify": False
}
DEFAULT_KERNELS_OPTS = {
    "args": False,
    "max_length": 10000,
    "temperature": 1
}


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


def from_json(sampler_json: dict):
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

        # Validate options
        for key in sampler_opts.keys():
            if key not in DEFAULT_SAMPLER_OPTS:
                raise clgen.UserError(
                    "Unsupported sampler option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_SAMPLER_OPTS.keys()))))
        for key in kernel_opts.keys():
            if key not in DEFAULT_KERNELS_OPTS:
                raise clgen.UserError(
                    "Unsupported kernels option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_KERNELS_OPTS.keys()))))

        # set properties
        self.sampler_opts = clgen.update(deepcopy(DEFAULT_SAMPLER_OPTS),
                                         sampler_opts)
        self.kernel_opts = clgen.update(deepcopy(DEFAULT_KERNELS_OPTS),
                                        kernel_opts)
        self.hash = self._hash(self.sampler_opts, self.kernel_opts)

        if self.sampler_opts["dynamic_checker"] and not cfg.USE_OPENCL:
            log.warning("dynamic checking requested, but OpenCL not available")
            self.sampler_opts["dynamic_checker"] = False

        def _start_text(args):
            if args == False:
                return "__kernel void A("
            else:
                return serialize_argspec(args)

        # seed to language model
        self.start_text = _start_text(self.kernel_opts["args"])

        # options to pass to preprocess_db()
        self.preprocess_opts = {
            "use_dynamic_checker": self.sampler_opts["dynamic_checker"],
            "use_gpuverify": self.sampler_opts["gpuverify"]
        }

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

        cache = Cache(fs.path("sampler", sampler_model_hash))

        # validate metadata against cache
        meta = self.to_json()
        if cache["META"]:
            cached_meta = clgen.load_json_file(cache["META"])
            if meta != cached_meta:
                raise clgen.InternalError("sampler metadata mismatch")
        else:
            clgen.write_json_file(cache.keypath("META"), meta)

        return cache

    def sample_iteration(self, model: Model, quiet: bool=False) -> None:
        """
        Run one sample iteration.

        Arguments:
            model (Model): CLgen model.
        """
        assert(isinstance(model, Model))

        cache = self.cache(model)

        tmppath = fs.path(cache.path,
                          "sampler-{pid}.tmp.cl".format(pid=system.PID))

        with open(tmppath, "w") as outfile:
            opts = {
                "output": outfile,
                "num_samples": self.sampler_opts["batch_size"],
                "temperature": self.kernel_opts["temperature"],
                "max_length": self.kernel_opts["max_length"],
                "seed_text": self.start_text,
                "quiet": quiet
            }
            model.sample(**opts)

        sys.stdout.flush()
        sys.stderr.flush()
        fetch.process_sample_file(
            cache["kernels.db"], tmppath,
            max_kernel_len=opts["max_length"], quiet=True)

        if self.sampler_opts["static_checker"]:
            preprocess.preprocess_db(cache["kernels.db"], **self.preprocess_opts)
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

        # properties
        max_kernels = self.sampler_opts["max_kernels"]
        has_max_kernels = max_kernels >= 0

        max_batches = self.sampler_opts["max_batches"]
        has_max_batches = max_batches >= 0


        batch_i = 0
        while True:
            # stop if we have enough kernels
            num_good_kernels = dbutil.num_good_kernels(cache["kernels.db"])
            if has_max_kernels and num_good_kernels >= max_kernels:
                return

            # stop if we've done enough batches
            if has_max_batches and batch_i >= max_batches:
                return

            batch_i += 1
            print("sample batch", batch_i, "...")

            self.sample_iteration(model, quiet=quiet)

            print()
            explore(self.cache(model)["kernels.db"])

        log.info("samples database:", cache["kernels.db"])

    def __repr__(self) -> str:
        """
        String representation.
        """
        hash = self.hash
        seed = self.start_text
        return "sampler[{hash}]: '{seed}'".format(**vars())

    def to_json(self) -> dict:
        """
        JSON representation.
        """
        return {
            "kernels": self.kernel_opts,
            "sampler": self.sampler_opts
        }

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Sampler):
            return False
        return rhs.hash == self.hash

    def __ne__(self, rhs) -> bool:
        return not self.__eq__(rhs)

    @staticmethod
    def from_json(sampler_json: dict):
        return from_json(sampler_json)
