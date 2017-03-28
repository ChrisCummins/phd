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
import numpy as np
import progressbar
import sys

from copy import deepcopy
from glob import glob, iglob
from io import StringIO
from labm8 import crypto
from labm8 import fs
from labm8 import jsonutil
from labm8 import lockfile
from labm8 import system
from labm8 import types
from threading import Condition, Event, Thread, Lock

import clgen
from clgen import clutil
from clgen import config as cfg
from clgen import dbutil
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen.explore import explore
from clgen.model import Model

# Default options used for sampler. Any values provided by the user will
# override these defaults.
DEFAULT_KERNELS_OPTS = {
    "args": None,
    "max_length": 10000,
    "seed": None,
    "temperature": 1
}
DEFAULT_SAMPLER_OPTS = {
    "min_samples": -1,
    "min_kernels": -1,
    "static_checker": True,
    "dynamic_checker": False,
    "gpuverify": False
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

    return Sampler(sampler_opts, kernel_opts)


class SampleProducer(Thread):
    def __init__(self, model: Model, start_text: str, condition: Condition,
                 queue: list, quiet: bool=False, **kernel_opts):
        super(SampleProducer, self).__init__()

        self.model = model
        self.start_text = start_text
        self.condition = condition
        self.queue = queue
        self.stop_signal = Event()
        self.kernel_opts = kernel_opts
        self.quiet = quiet

    def run(self):
        model = self.model
        max_length = self.kernel_opts["max_length"]
        temperature = self.kernel_opts["temperature"]

        if model.lock.islocked:  # model is locked during training
            raise lockfile.UnableToAcquireLockError(self.lock)

        tf = model._init_tensorflow(infer=True)

        # seed RNG
        np.random.seed(self.kernel_opts["seed"])
        tf.set_random_seed(self.kernel_opts["seed"])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model.cache.path)

            assert(ckpt)
            assert(ckpt.model_checkpoint_path)

            saver.restore(sess, ckpt.model_checkpoint_path)

            def weighted_pick(weights, temperature):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return int(np.searchsorted(t, np.random.rand(1) * s))

            def update_bracket_depth(text, started: bool=False, depth: int=0):
                """ calculate function block depth """
                for char in text:
                    if char == '{':
                        depth += 1
                        started = True
                    elif char == '}':
                        depth -= 1

                return started, depth

            init_started, init_depth = update_bracket_depth(self.start_text)

            while not self.stop_requested:
                buf = StringIO()
                started, depth = init_started, init_depth

                state = sess.run(model.cell.zero_state(1, tf.float32))

                seed_tensor = model.corpus.atomizer.atomize(self.start_text)
                for index in seed_tensor[:-1]:
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {model.input_data: x, model.initial_state: state}
                    [state] = sess.run([model.final_state], feed)

                buf.write(self.start_text)
                if not self.quiet:
                    sys.stdout.write("\n\n/* ==== START SAMPLE ==== */\n\n")
                    sys.stdout.write(self.start_text)
                    sys.stdout.flush()

                index = seed_tensor[-1]

                for _ in range(max_length):
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {model.input_data: x, model.initial_state: state}
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              feed)
                    p = probs[0]

                    # sample distribution to pick next:
                    index = weighted_pick(p, temperature)
                    # alternatively, select most probable:
                    # index = np.argmax(p)

                    atom = model.corpus.atomizer.deatomize([index])
                    buf.write(atom)
                    if not self.quiet:
                        sys.stdout.write(atom)

                    # update function block depth
                    started, depth = update_bracket_depth(atom, started, depth)

                    # stop sampling if depth <= 0
                    if started and depth <= 0:
                        break

                # submit sample to processing queue
                self.condition.acquire()
                self.queue.append(buf.getvalue())

                self.condition.notify()
                self.condition.release()

            if not self.quiet:
                sys.stdout.write('\n\n')

    def stop(self):
        self.stop_signal.set()

    @property
    def stop_requested(self):
        return self.stop_signal.isSet()


class SampleConsumer(Thread):
    """ handle generated samples """
    def __init__(self, db_path: str, sampler: SampleProducer,
                 condition: Condition, queue: list, quiet: bool=False,
                 **sampler_opts):
        """
        Arguments:
            db_path (str): Path to samples database.
            sampler (SampleProducer):
        """
        super(SampleConsumer, self).__init__()

        self.db_path = db_path
        self.sampler = sampler
        self.condition = condition
        self.queue = queue
        self.sampler_opts = sampler_opts
        self.quiet = quiet

        # properties
        min_kernels = self.sampler_opts["min_kernels"]
        has_min_kernels = min_kernels >= 0

        min_samples = self.sampler_opts["min_samples"]
        has_min_samples = min_samples >= 0

        # Determine termination criteria
        if has_min_kernels and has_min_samples:
            self.term_condition = self.min_kernels_and_samples_cond
            self.max_i = self.sampler_opts["min_kernels"]
            self.progress = self.min_kernels_progress
        elif has_min_kernels:
            self.term_condition = self.min_kernels_cond
            self.max_i = self.sampler_opts["min_kernels"]
            self.progress = self.min_kernels_progress
        elif has_min_samples:
            self.term_condition = self.min_samples_cond
            self.max_i = self.sampler_opts["min_samples"]
            self.progress = self.min_samples_progress
        else:
            self.term_condition = self.null_cond
            self.max_i = progressbar.UnknownLength
            self.progress = self.min_samples_progress

    def min_kernels_and_samples_cond(self):
        return self.min_kernels_cond() and self.min_samples_cond()

    def min_kernels_cond(self):
        return self.min_kernels_progress() >= self.sampler_opts["min_kernels"]

    def min_samples_cond(self):
        return (dbutil.num_rows_in(self.db_path, "ContentFiles") >=
                self.sampler_opts["min_samples"])

    def null_cond(self):
        return False

    def min_kernels_progress(self):
        return min(dbutil.num_good_kernels(self.db_path),
                   self.sampler_opts["min_kernels"])

    def min_samples_progress(self):
        return min(dbutil.num_rows_in(self.db_path, "ContentFiles"),
                   self.sampler_opts["min_samples"])

    def run(self) -> None:
        i = dbutil.num_rows_in(self.db_path, "ContentFiles")

        if self.quiet:
            bar = progressbar.ProgressBar(max_value=self.max_i)
            bar.update(self.progress())

        while True:
            # get the next sample
            self.condition.acquire()
            if not self.queue:
                self.condition.wait()
            sample = self.queue.pop(0)
            self.condition.release()

            kernels = clutil.get_cl_kernels(sample)
            ids = [crypto.sha1_str(k) for k in kernels]

            if self.sampler_opts["static_checker"]:
                preprocess_opts = {
                    "use_shim": False,
                    "use_dynamic_checker": self.sampler_opts["dynamic_checker"],
                    "use_gpuverify": self.sampler_opts["gpuverify"]
                }
                pp = [preprocess.preprocess_for_db(k, **preprocess_opts)
                      for k in kernels]

            db = dbutil.connect(self.db_path)
            c = db.cursor()

            # insert raw samples
            for kid, src in zip(ids, kernels):
                dbutil.sql_insert_dict(c, "ContentFiles",
                                       {"id": kid, "contents": src},
                                       replace_existing=True)

            # insert preprocessed samples
            if self.sampler_opts["static_checker"]:
                for kid, (status, src) in zip(ids, pp):
                    dbutil.sql_insert_dict(c, "PreprocessedFiles", {
                        "id": kid, "status": status, "contents": src
                    }, replace_existing=True)

            c.close()
            db.commit()
            db.close()

            # update progress bar
            if self.quiet:
                bar.update(self.progress())

            # determine if we are done sampling
            if self.term_condition():
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
        def _hash(sampler_opts: dict, kernel_opts: dict) -> str:
            # we don't consider the number of samples in the ID
            sampler_opts = deepcopy(sampler_opts)
            del sampler_opts["min_samples"]
            del sampler_opts["min_kernels"]

            checksum_data = sorted(
                [str(x) for x in sampler_opts.values()] +
                [str(x) for x in kernel_opts.values()])
            string = "".join([str(x) for x in checksum_data])
            return crypto.sha1_str(string)

        def _start_text(args):
            if args is None:
                return "__kernel void A("
            else:
                return serialize_argspec(args)

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
        self.sampler_opts = types.update(deepcopy(DEFAULT_SAMPLER_OPTS),
                                         sampler_opts)
        self.kernel_opts = types.update(deepcopy(DEFAULT_KERNELS_OPTS),
                                        kernel_opts)

        self.hash = _hash(self.sampler_opts, self.kernel_opts)

        if self.sampler_opts["dynamic_checker"] and not cfg.USE_OPENCL:
            log.warning("dynamic checking requested, but OpenCL not available")
            self.sampler_opts["dynamic_checker"] = False

        self.start_text = _start_text(self.kernel_opts["args"])

        # options to pass to preprocess_db()
        self.preprocess_opts = {
            "use_dynamic_checker": self.sampler_opts["dynamic_checker"],
            "use_gpuverify": self.sampler_opts["gpuverify"]
        }

    def cache(self, model: Model):
        """
        Return sampler cache.

        Arguments:
            model (Model): CLgen model.

        Returns:
            labm8.FSCache: Cache.
        """
        sampler_model_hash = crypto.sha1_str(self.hash + model.hash)

        cache = clgen.mkcache("sampler", sampler_model_hash)

        # validate metadata against cache
        meta = deepcopy(self.to_json())
        del meta["sampler"]["min_samples"]
        del meta["sampler"]["min_kernels"]

        if cache.get("META"):
            cached_meta = jsonutil.read_file(cache["META"])
            if meta != cached_meta:
                raise clgen.InternalError("sampler metadata mismatch")
        else:
            jsonutil.write_file(cache.keypath("META"), meta)

        return cache

    def sample(self, model: Model, quiet: bool=False) -> None:
        """
        Sample CLgen model.

        Arguments:
            model (Model): CLgen model.
        """
        cache = self.cache(model)

        # create samples database if it doesn't exist
        if not cache.get("kernels.db"):
            tmp_kernels_db = cache.keypath("kernels.tmp.db")
            dbutil.create_db(tmp_kernels_db)
            cache["kernels.db"] = tmp_kernels_db

        # producer-consumer queue
        queue = []
        lock = Lock()
        condition = Condition()

        sampler = SampleProducer(model, self.start_text, condition, queue,
                                 quiet=quiet, **self.kernel_opts)
        sampler.start()

        consumer = SampleConsumer(cache["kernels.db"], sampler, condition,
                                  queue, quiet=quiet, **self.sampler_opts)
        consumer.start()

        sampler.join()
        consumer.join()
        print()
        explore(cache["kernels.db"])

    @property
    def min_samples(self):
        return self.sampler_opts["min_samples"]

    @property
    def num_samples(self):
        return dbutil.num_rows_in(self.db_path, "ContentFiles")

    @property
    def min_kernels(self):
        return self.sampler_opts["min_kernels"]

    @property
    def num_good_kernels(self):
        return dbutil.num_good_kernels(self.db_path)

    def to_json(self) -> dict:
        """
        JSON representation.
        """
        return {
            "kernels": self.kernel_opts,
            "sampler": self.sampler_opts
        }

    def __repr__(self) -> str:
        """
        String representation.
        """
        hash = self.hash
        seed = self.start_text
        return "sampler[{hash}]: '{seed}'".format(**vars())

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Sampler):
            return False
        return rhs.hash == self.hash

    def __ne__(self, rhs) -> bool:
        return not self.__eq__(rhs)

    @staticmethod
    def from_json(sampler_json: dict):
        return from_json(sampler_json)
