#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
from datetime import datetime
from io import StringIO
from queue import Queue
from threading import Event, Thread
from time import time
from typing import List, Union

import numpy as np
import progressbar
from tensorflow.python.framework.errors import InvalidArgumentError as \
  TensorFlowInvalidArgumentError

import deeplearning.clgen.clgen.errors
from deeplearning.clgen import clgen
from deeplearning.clgen import dbutil
from deeplearning.clgen import log
from lib.labm8 import crypto
from lib.labm8 import jsonutil
from lib.labm8 import lockfile
from lib.labm8 import types


# Default options used for sampler. Any values provided by the user will
# override these defaults.
DEFAULT_KERNELS_OPTS = {"language": None,  # note language must be explicitly provided
                        "args": None, "start_text": None, "max_length": 10000, "seed": None,
                        "temperature": 1}
DEFAULT_SAMPLER_OPTS = {
  "created": {"author": clgen.get_default_author(), "date": str(datetime.now()),
              "version": clgen.version(), }, "min_samples": -1, "min_kernels": -1,
  # FIXME(polyglot): remove static checker and gpuverify
  "static_checker": True, "gpuverify": False}


def serialize_opencl_argspec(args: List[str]) -> str:
  """
  Serializes an argument spec to a kernel prototype.

  Parameters
  ----------
  args : List[str]
      Argument specification.

  Returns
  -------
  str
      Kernel prototype.
  """
  names = map(chr, range(97, 97 + len(args)))
  strings = [arg + " " + name for arg, name in zip(args, names)]
  return "__kernel void A({args}) {{".format(args=", ".join(strings))


class SampleProducer(Thread):
  def __init__(self, model: clgen.Model, start_text: str, queue: Queue, **kernel_opts):
    super(SampleProducer, self).__init__()

    self.model = model
    self.start_text = start_text
    self.queue = queue
    self.stop_signal = Event()
    self.kernel_opts = kernel_opts
    self.sample_header = "\n\n" + clgen.format_as_comment(model.corpus.language,
                                                          "==== START SAMPLE ====") + "\n\n"

  def run(self) -> None:
    model = self.model
    batch_size = self.model.corpus.batch_size
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

      assert (ckpt)
      assert (ckpt.model_checkpoint_path)

      saver.restore(sess, ckpt.model_checkpoint_path)

      def weighted_pick(weights, temperature):
        """
        requires that all probabilities are >= 0, i.e.:
          assert all(x >= 0 for x in weights)
        See: https://github.com/ChrisCummins/clgen/issues/120
        """
        t = np.cumsum(weights)
        s = np.sum(weights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

      def get_bracket_depth(text: str, depth: int = 0) -> int:
        """ calculate function block depth """
        # FIXME(polyglot): support multiple sample termination criteria
        depth += text.count("{")
        started = depth > 0
        depth -= text.count("}")
        return started, depth

      init_started, init_depth = get_bracket_depth(self.start_text)
      atomize = model.corpus.atomizer.atomize
      deatomize = model.corpus.atomizer.deatomize

      while not self.stop_requested:
        buf = [StringIO() for _ in range(batch_size)]
        depth = [init_depth] * batch_size
        started = [init_started] * batch_size
        running = [True] * batch_size

        state = sess.run(model.cell.zero_state(batch_size, tf.float32))
        indices = np.zeros((batch_size, 1))

        seed_tensor = atomize(self.start_text)
        for symbol in seed_tensor[:-1]:
          indices[:] = symbol
          feed = {model.input_data: indices, model.initial_state: state}
          [state] = sess.run([model.final_state], feed)

        for item in range(batch_size):
          buf[item].write(self.start_text)

        indices[:] = seed_tensor[-1]

        for _ in range(max_length):
          feed = {model.input_data: indices, model.initial_state: state}

          try:
            [probs, state] = sess.run([model.probs, model.final_state], feed)
          except TensorFlowInvalidArgumentError:
            log.warning("sampling error")
            self.run()

          # sample distribution to pick next symbol:
          indices[:, 0] = [weighted_pick(p, temperature) for p in probs]

          for item in range(batch_size):
            if not running[item]:
              continue

            # In case of decoding error, start sampling again:
            try:
              atom = deatomize([indices[item, 0]])
            except clgen.VocabError:
              log.warning("deatomizing error")
              self.run()

            buf[item].write(atom)
            # update function block depth
            _started, depth[item] = get_bracket_depth(atom, depth=depth[item])
            started[item] |= _started  # you can't "unset" the started state
            # determine whether to keep sampling:
            _running = not started[item] or (started[item] and depth[item] > 0)
            running[item] = _running

            # submit sample to processing queue
            if not _running:
              text = buf[item].getvalue()
              self.queue.put(text)
              if log.is_verbose():
                sys.stdout.write(self.sample_header)
                sys.stdout.write(text)
                sys.stdout.flush()

          # start a new batch if there's nothing left running
          if not any(running):
            break

      if log.is_verbose():
        sys.stdout.write('\n\n')

  def stop(self) -> None:
    self.stop_signal.set()

  @property
  def stop_requested(self) -> bool:
    return self.stop_signal.isSet()


class SampleConsumer(Thread):
  """ handle generated samples """

  def __init__(self, db_path: str, producer: SampleProducer, sampler, cache, queue: Queue,
               **sampler_opts):
    """
    Construct a sample consumer.

    Parameters
    ----------
    db_path : str
        Path to samples database.
    producer : SampleProducer
        Sample producer thread.
    sampler : Sampler
        Host sampler.
    condition : Condition
        For locking.
    queue : Queue
        output result queue.
    **sampler_opts
        Sampler options.
    """
    super(SampleConsumer, self).__init__()

    self.db_path = db_path
    self.producer = producer
    self.sampler = sampler
    self.cache = cache
    self.queue = queue
    self.sampler_opts = sampler_opts

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
      self.progress = self.null_progress

  def min_kernels_and_samples_cond(self) -> bool:
    return self.min_kernels_cond() and self.min_samples_cond()

  def min_kernels_cond(self) -> bool:
    return self.min_kernels_progress() >= self.sampler_opts["min_kernels"]

  def min_samples_cond(self) -> bool:
    return (dbutil.num_rows_in(self.db_path, "ContentFiles") >= self.sampler_opts["min_samples"])

  def null_cond(self) -> bool:
    return False

  def min_kernels_progress(self) -> int:
    return min(dbutil.num_good_kernels(self.db_path), self.sampler_opts["min_kernels"])

  def min_samples_progress(self) -> int:
    return min(dbutil.num_rows_in(self.db_path, "ContentFiles"), self.sampler_opts["min_samples"])

  def null_progress(self) -> int:
    return dbutil.num_rows_in(self.db_path, "ContentFiles")

  def run(self) -> None:
    i = dbutil.num_rows_in(self.db_path, "ContentFiles")

    if not log.is_verbose():
      bar = progressbar.ProgressBar(max_value=self.max_i)
      bar.update(self.progress())

    try:
      while True:
        sample_time = time()

        # Block while waiting for a new sample to come in:
        sample = self.queue.get(timeout=120).strip()

        # Compute the sample ID:
        kid = crypto.sha1_str(sample)

        # Add the new sample to the database:
        db = dbutil.connect(self.db_path)
        c = db.cursor()
        dbutil.sql_insert_dict(c, "ContentFiles", {"id": kid, "contents": sample},
                               ignore_existing=True)
        c.close()
        db.commit()
        db.close()

        # update progress bar
        progress = self.progress()
        if not log.is_verbose():
          bar.update(progress)

        sample_time = time() - sample_time
        self.sampler.stats["progress"] = progress
        self.sampler.stats["time"] += sample_time
        self.sampler._flush_meta(self.cache)

        # determine if we are done sampling
        if self.term_condition():
          self.producer.stop()
          return
    finally:  # always kill the sampler thread
      print()
      self.producer.stop()


class Sampler(object):
  """
  CLgen sampler for models.

  Please note sampler instances should be treated as immutable. Upon
  instantiation, a sampler's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, sampler_opts: dict, kernel_opts: dict):
    """
    Instantiate a sampler.

    Parameters
    ----------
    sampler_opts : dict
        Sampler options.
    kernel_opts : dict
        Kernel options.
    """

    def _hash(sampler_opts: dict, kernel_opts: dict) -> str:
      # we don't consider the number of samples in the ID
      sampler_opts = deepcopy(sampler_opts)
      del sampler_opts["min_samples"]
      del sampler_opts["min_kernels"]
      del sampler_opts["created"]

      checksum_data = sorted(
        [str(x) for x in sampler_opts.values()] + [str(x) for x in kernel_opts.values()])
      string = "".join([str(x) for x in checksum_data])
      return crypto.sha1_str(string)

    # FIXME(polyglot):
    def _start_text(lang: clgen.Language, args: Union[List[str], None], start_text: str):
      if lang == clgen.Language.OPENCL:
        if args is None:
          return "__kernel void A("
        else:
          return serialize_opencl_argspec(args)
      else:
        return start_text or ""

    assert (type(sampler_opts) is dict)
    assert (type(kernel_opts) is dict)

    # Validate options
    for key in sampler_opts.keys():
      if key not in DEFAULT_SAMPLER_OPTS:
        raise deeplearning.clgen.clgen.errors.UserError(
          "Unsupported sampler option '{}'. Valid keys: {}".format(key, ','.join(
            sorted(DEFAULT_SAMPLER_OPTS.keys()))))
    for key in kernel_opts.keys():
      if key not in DEFAULT_KERNELS_OPTS:
        raise deeplearning.clgen.clgen.errors.UserError(
          "Unsupported kernels option '{}'. Valid keys: {}".format(key, ','.join(
            sorted(DEFAULT_KERNELS_OPTS.keys()))))

    # set properties
    self.sampler_opts = types.update(deepcopy(DEFAULT_SAMPLER_OPTS), sampler_opts)
    self.kernel_opts = types.update(deepcopy(DEFAULT_KERNELS_OPTS), kernel_opts)

    self.hash = _hash(self.sampler_opts, self.kernel_opts)

    self.language = clgen.Language.from_str(kernel_opts.get("language"))

    self.start_text = _start_text(self.language, self.kernel_opts.get("args", []),
                                  self.kernel_opts.get("start_text", ""))
    # pop "start_text" option
    del self.kernel_opts["start_text"]

    # options to pass to preprocess_db()
    self.preprocess_opts = {"use_gpuverify": self.sampler_opts["gpuverify"]}

  def cache(self, model: clgen.Model):
    """
    Return sampler cache.

    Parameters
    ----------
    model : clgen.Model
        CLgen model.

    Returns
    -------
    labm8
        FSCache: Cache.
    """
    sampler_model_hash = crypto.sha1_str(self.hash + model.hash)

    cache = clgen.mkcache("sampler", f"{self.language}-{sampler_model_hash}")

    # validate metadata against cache
    self.stats = {"time": 0, "progress": 0}
    meta = deepcopy(self.to_json())
    if cache.get("META"):
      cached_meta = jsonutil.read_file(cache["META"])

      if "stats" in cached_meta:
        self.stats = cached_meta["stats"]
        del cached_meta["stats"]

      if "created" in cached_meta["sampler"]:
        del cached_meta["sampler"]["created"]
      del meta["sampler"]["created"]

      if "min_samples" in cached_meta["sampler"]:
        del cached_meta["sampler"]["min_samples"]
      del meta["sampler"]["min_samples"]

      if "min_kernels" in cached_meta["sampler"]:
        del cached_meta["sampler"]["min_kernels"]
      del meta["sampler"]["min_kernels"]

      if meta != cached_meta:
        raise deeplearning.clgen.clgen.errors.InternalError("sampler metadata mismatch")
    else:
      self._flush_meta(cache)

    return cache

  def _flush_meta(self, cache):
    jsonutil.write_file(cache.keypath("META"), self.to_json(cache))

  def sample(self, model: clgen.Model) -> None:
    """
    Sample CLgen model.

    Parameters
    ----------
    model : clgen.Model
        CLgen model.
    """
    cache = self.cache(model)

    # create samples database if it doesn't exist
    if not cache.get("kernels.db"):
      tmp_kernels_db = cache.keypath("kernels.tmp.db")
      dbutil.create_db(tmp_kernels_db)
      cache["kernels.db"] = tmp_kernels_db

    # producer-consumer queue
    queue = Queue(maxsize=128)

    log.info("sampling", self)

    sampler = SampleProducer(model, self.start_text, queue, **self.kernel_opts)
    sampler.start()

    consumer = SampleConsumer(cache["kernels.db"], sampler, self, cache, queue, **self.sampler_opts)
    consumer.start()

    sampler.join()
    consumer.join()

    clgen.explore(cache["kernels.db"])

  @property
  def shorthash(self) -> str:
    return clgen._shorthash(self.hash, clgen.cachepath("sampler"))

  @property
  def min_samples(self) -> int:
    return self.sampler_opts["min_samples"]

  @property
  def num_samples(self) -> int:
    return dbutil.num_rows_in(self.db_path, "ContentFiles")

  @property
  def min_kernels(self) -> int:
    return self.sampler_opts["min_kernels"]

  @property
  def num_good_kernels(self) -> int:
    return dbutil.num_good_kernels(self.db_path)

  def to_json(self, cache=None) -> dict:
    """
    JSON representation.

    Returns
    -------
    dict
        JSON specification.
    """
    d = {"kernels": self.kernel_opts, "sampler": self.sampler_opts}

    if cache:
      d["stats"] = self.stats

    return d

  def __repr__(self) -> str:
    """
    String representation.
    """
    return f"sampler[{self.shorthash}]: '{self.start_text}'"

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Sampler):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)

  @staticmethod
  def from_json(sampler_json: dict) -> 'Sampler':
    """
    Instantiate sampler from JSON.

    Parameters
    ----------
    sampler_json : dict
        JSON data.

    Returns
    -------
    Sampler
        Instantiate sampler.
    """
    unrecognized_keys = (set(sampler_json.keys()) - set(["sampler", "kernels"]))
    if unrecognized_keys:
      raise deeplearning.clgen.clgen.errors.UserError(
        "unrecognized sampler JSON options '{}'".format(
          ",".join(["'{}'".format(key) for key in unrecognized_keys])))

    sampler_opts = sampler_json.get("sampler", {})
    kernel_opts = sampler_json.get("kernels", {})

    return Sampler(sampler_opts, kernel_opts)
