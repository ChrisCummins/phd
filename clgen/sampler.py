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
from threading import Condition, Event, Thread, Lock

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
    "min_kernels": -1,
    "min_samples": -1,
    "static_checker": True,
    "dynamic_checker": False,
    "gpuverify": False
}
DEFAULT_KERNELS_OPTS = {
    "args": False,
    "max_length": 10000,
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

class SampleProducer(Thread):
    def __init__(self, model, condition, queue, **kernel_opts):
        super(SampleProducer, self).__init__()
        self.model = model
        self.condition = condition
        self.queue = queue
        self.stop_signal = Event()
        self.kernel_opts = kernel_opts

    def run(self):
        while not self.stop_requested:
            # TODO: create sample
            # "max_length": self.kernel_opts["max_length"],
            # "seed_text": self.start_text,

            self.condition.acquire()
            self.queue.append(sample)
            self.condition.notify()
            self.condition.release()

    def stop(self):
        self.stop_signal.set()

    @property
    def stop_requested(self):
        return self._stop_signal.isSet()


class SampleConsumer(Thread):
    def __init__(self, db_path: str, sampler: SampleProducer, condition, queue,
                 **sampler_opts):
        super(SampleConsumer, self).__init__()

        self.db_path = db_path
        self.sampler = sampler
        self.condition = condition
        self.queue = queue
        self.sampler_opts = sampler_opts

        # properties
        min_kernels = sampler_opts["min_kernels"]
        has_min_kernels = min_kernels >= 0

        min_samples = sampler_opts["min_samples"]
        has_min_samples = min_batches >= 0

        def min_kernels_cond(i):
            return dbutil.num_good_kernels(self.db_path) >= min_kernels
        def min_samples_cond(i):
            return i >= min_samples

        # Determine termination criteria
        if has_min_kernels and has_min_samples:
            term_condition = lambda i: (min_kernels_cond(i) and
                                        min_samples_cond(i))
        if has_min_kernels:
            term_condition = min_kernels_cond
        elif has_min_samples:
            term_condition = min_samples_cond
        else:
            term_condition = lambda i: False

        self.term_condition = term_condition

    def run(self):
        # bar = progressbar.ProgressBar(max_value=self.numjobs)

        i = 0

        while True:
            # get the next sample
            self.condition.acquire()
            if not self.queue:
                self.condition.wait()
            sample = self.queue.pop(0)
            self.condition.release()

            i += 1

            kernels = get_cl_kernels(sample)
            ids = [crypto.sha1_str(k) for k in kernels]

            if self.sample_opts["static_checker"]:
                preprocess_opts = {
                    "use_shim": False,
                    "use_gpuverify": False
                }
                pp = [preprocess_for_db(k, **preprocess_opts) in kernels]

            db = dbutil.connect(self.db_path)
            c = db.cursor()

            for kid, src in zip(ids, kernels):
                dbutil.sql_insert_dict(c, "ContentFiles",
                                       {"id": kid, "contents": src})

            if sampler_opts["static_checker"]:
                for kid, (status, src) in zip(ids, pp):
                    dbutil.sql_insert_dict(c, "PreprocessedFiles", {
                        "id": kid, "status": status, "contents": src
                    })

            c.close()
            db.commit()
            db.close()

            if self.term_condition(i):
                # signal the sampler to stop
                self.sampler.stop()
                return

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

        # producer-consumer queue
        queue = []
        lock = Lock()
        condition = Condition()

        sampler = SampleProducer(model, condition, queue, **self.kernel_opts)
        sampler.start()

        consumer = SampleConsumer(sampler, condition, queue, **self.sample_opts)
        consumer.start()

        producer.join()
        consumer.join()
        print()
        explore(cache["kernels.db"])


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
