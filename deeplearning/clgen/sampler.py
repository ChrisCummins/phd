"""Samplers for CLgen language models."""
import sys
import typing
from io import StringIO
from queue import Queue
from threading import Event, Thread
from time import time

import numpy as np
import progressbar
from absl import logging
from tensorflow.python.framework import errors as tf_errors

from deeplearning.clgen import cache
from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen import explore
from deeplearning.clgen import languages
from deeplearning.clgen import model
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import cache as labcache
from lib.labm8 import crypto
from lib.labm8 import jsonutil
from lib.labm8 import lockfile


class Sampler(object):
  """CLgen sampler for models.

  Please note sampler instances should be treated as immutable. Upon
  instantiation, a sampler's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: sampler_pb2.Sampler):
    """Instantiate a sampler.

    Args:
      config: A Sampler message.
    """
    self.config = sampler_pb2.Sampler()
    self.config.CopyFrom(config)
    self.hash = self._ComputeHash(self.config)
    if not config.start_text:
      raise errors.UserError('Sampler.start_text not set')
    if config.batch_size < 1:
      raise errors.UserError('Sampler.batch_size must be >= 1')

  @staticmethod
  def _ComputeHash(config: sampler_pb2.Sampler) -> str:
    """Compute sampler hash.

    The hash is computed from the serialized representation of the config
    proto.
    """
    return crypto.sha1(config.SerializeToString())

  def cache(self, model_: model.Model) -> labcache.FSCache:
    """Return sampler cache.

    Args:
      model: The CLgen model being sampler.

    Returns:
      A FSCache cache.
    """
    sampler_model_hash = crypto.sha1_list([self.hash, model_.hash])
    cache_ = cache.mkcache('sampler', f'{model_.corpus.language}-'
                                      f'{sampler_model_hash}')
    self.stats = {'time': 0, 'progress': 0}
    self.meta = {}
    if cache_.get('META'):
      cached_meta = jsonutil.read_file(cache_['META'])
      if 'stats' in cached_meta:
        self.stats = cached_meta['stats']
        del cached_meta['stats']
      if self.meta != cached_meta:
        raise errors.InternalError('sampler metadata mismatch')
      self.meta['stats'] = self.stats
    else:
      self._FlushMeta(cache_)
    return cache_

  def _FlushMeta(self, cache_):
    jsonutil.write_file(cache_.keypath('META'), self.meta)

  def Sample(self, model_: model.Model) -> None:
    """Sample CLgen model.

    Args:
      model: The CLgen model to sample.
    """
    cache_ = self.cache(model_)
    # Create samples database if it doesn't exist.
    if not cache_.get('kernels.db'):
      tmp_kernels_db = cache_.keypath('kernels.tmp.db')
      dbutil.create_db(tmp_kernels_db)
      cache_['kernels.db'] = tmp_kernels_db
    # Producer-consumer queue.
    queue = Queue(maxsize=128)
    logging.info('sampling %s', self)
    sampler = SampleProducer(model_, self.config, queue)
    sampler.start()
    consumer = SampleConsumer(cache_['kernels.db'], sampler, self, cache_,
                              queue)
    consumer.start()
    sampler.join()
    consumer.join()
    explore.explore(cache_['kernels.db'])

  @property
  def shorthash(self) -> str:
    return cache.ShortHash(self.hash, cache.cachepath('sampler'))

  @property
  def min_samples(self) -> int:
    return self.config.min_num_samples

  @property
  def num_samples(self) -> int:
    return dbutil.num_rows_in(self.db_path, 'ContentFiles')

  def __repr__(self) -> str:
    """String representation."""
    return f'sampler[{self.shorthash}]: "{self.config.start_text}"'

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Sampler):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


class SampleProducer(Thread):
  def __init__(self, model_: model.Model, sampler_config: sampler_pb2.Sampler,
               queue: Queue):
    super(SampleProducer, self).__init__()
    self.model = model_
    self.sampler_config = sampler_config
    self.queue = queue
    self.stop_signal = Event()
    self.sample_header = '\n\n' + languages.format_as_comment(
      self.model.corpus.language, '==== START SAMPLE ====') + '\n\n'

    # Determine the termination criteria.
    self.max_length = -1
    self.special_token_left = None
    self.special_token_right = None
    for criterion in self.sampler_config.termination_criteria:
      if criterion.HasField('maxlen'):
        self.max_length = criterion.maxlen.maximum_tokens_in_sample
        if not criterion.maxlen.include_start_text_in_maximum:
          self.max_length += len(self.sampler_config.start_text)
      elif criterion.HasField('symtok'):
        self.symmetrical_token_left = criterion.symtok.depth_increase_token
        self.symmetrical_token_right = criterion.symtok.depth_decrease_token
    self.has_max_length = self.max_length > 0
    self.has_symmetrical_tokens = (
        self.special_token_left and self.special_token_right)

  def run(self) -> None:
    batch_size = self.sampler_config.batch_size

    # Fail if the model is locked. Models are locked during training.
    if self.model.lock.islocked:
      raise lockfile.UnableToAcquireLockError(self.lock)

    tf = self.model.InitAndGetTensorflow(infer=True)

    # Seed the RNG.
    np.random.seed(self.sampler_config.seed)
    tf.set_random_seed(self.sampler_config.seed)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(tf.global_variables())
      checkpoint = tf.train.get_checkpoint_state(self.model.cache.path)
      # Sanity checks.
      assert checkpoint
      assert checkpoint.model_checkpoint_path
      saver.restore(sess, checkpoint.model_checkpoint_path)

      def WeightedPick(weights):
        """Make a weighted choice.

        Requires that all probabilities are >= 0, i.e.:
          assert all(x >= 0 for x in weights)
        See: https://github.com/ChrisCummins/clgen/issues/120
        """
        t = np.cumsum(weights)
        s = np.sum(weights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

      def GetSymmetricalTokenDepth(text: str, depth: int) -> typing.Tuple[
        int, int]:
        """Calculate the sample depth for symmetrical tokens."""
        depth += text.count(self.symmetrical_token_left)
        is_started = depth > 0
        depth -= text.count(self.symmetrical_token_right)
        return is_started, depth

      if self.has_symmetrical_tokens:
        init_started, init_depth = GetSymmetricalTokenDepth(
          self.sampler_config.start_text, 0)
      atomize = self.model.corpus.atomizer.AtomizeString
      deatomize = self.model.corpus.atomizer.DeatomizeIndices

      while not self.stop_requested:
        buf = [StringIO() for _ in range(batch_size)]
        if self.has_symmetrical_tokens:
          depth = [init_depth] * batch_size
          started = [init_started] * batch_size
        running = [True] * batch_size

        state = sess.run(self.model.cell.zero_state(batch_size, tf.float32))
        indices = np.zeros((batch_size, 1))

        seed_tensor = atomize(self.sampler_config.start_text)
        for symbol in seed_tensor[:-1]:
          indices[:] = symbol
          feed = {self.model.input_data: indices,
                  self.model.initial_state: state}
          [state] = sess.run([self.model.final_state], feed)
        for item in range(batch_size):
          buf[item].write(self.sampler_config.start_text)
        indices[:] = seed_tensor[-1]
        i = 0
        while True:
          feed = {self.model.input_data: indices,
                  self.model.initial_state: state}
          try:
            [probs, state] = sess.run(
              [self.model.probs, self.model.final_state], feed)
          except tf_errors.InvalidArgumentError:
            logging.warning('sampling error')
            self.run()
          # Sample distribution to pick next symbols:
          indices[:, 0] = [WeightedPick(p) for p in probs]

          for item in range(batch_size):
            if not running[item]:
              continue

            # In case of decoding error, start sampling again:
            # try:
            atom = deatomize([indices[item, 0]])
            # except errors.VocabError:
            #   logging.warning('deatomizing error')
            #   self.run()
            buf[item].write(atom)
            # Update symmetrical character depths.
            if self.has_symmetrical_tokens:
              _started, depth[item] = GetSymmetricalTokenDepth(atom,
                                                               depth=depth[
                                                                 item])
              started[item] |= _started  # You can't 'unset' the started state.
              running[item] = (
                  not started[item] or (started[item] and depth[item] > 0))
            # Update max length.
            if self.has_max_length and i >= self.max_length:
              running[item] = False
            # Submit sample to processing queue.
            if not running[item]:
              text = buf[item].getvalue()
              self.queue.put(text)
              if logging.get_verbosity() == logging.DEBUG:
                sys.stdout.write(self.sample_header)
                sys.stdout.write(text)
                sys.stdout.flush()
          # Start a new batch if there's nothing left running.
          if not any(running):
            break
          i += 1

      if logging.get_verbosity() == logging.DEBUG:
        sys.stdout.write('\n\n')

  def stop(self) -> None:
    self.stop_signal.set()

  @property
  def stop_requested(self) -> bool:
    return self.stop_signal.isSet()


class SampleConsumer(Thread):
  """Handle generated samples."""

  def __init__(self, db_path: str, producer: SampleProducer, sampler: Sampler,
               cache_: labcache.FSCache, queue: Queue):
    """Instantiate a SampleConsumer.

    Args:
      db_path: Path to the sampler database.
      producer: A SampleProducer thread.
      sampler: The host Sampler instance.
      cache_: The sampler cache.
      queue: A shared queue.
    """
    super(SampleConsumer, self).__init__()
    self.db_path = db_path
    self.producer = producer
    self.sampler = sampler
    self.cache = cache_
    self.queue = queue
    min_samples = self.sampler.config.min_num_samples
    has_min_samples = min_samples >= 0
    # Determine termination criteria.
    if has_min_samples:
      self.term_condition = self.min_samples_cond
      self.max_i = min_samples
      self.progress = self.min_samples_progress
    else:
      self.term_condition = self.null_cond
      self.max_i = progressbar.UnknownLength
      self.progress = self.null_progress

  def min_samples_cond(self) -> bool:
    return (dbutil.num_rows_in(self.db_path, 'ContentFiles') >= self.max_i)

  def null_cond(self) -> bool:
    return False

  def min_samples_progress(self) -> int:
    return min(dbutil.num_rows_in(self.db_path, 'ContentFiles'), self.max_i)

  def null_progress(self) -> int:
    return dbutil.num_rows_in(self.db_path, 'ContentFiles')

  def run(self) -> None:
    i = dbutil.num_rows_in(self.db_path, 'ContentFiles')

    if not logging.get_verbosity() == logging.DEBUG:
      bar = progressbar.ProgressBar(max_value=self.max_i)
      bar.update(self.progress())

    try:
      while True:
        sample_time = time()
        # Block while waiting for a new sample to come in.
        sample = self.queue.get(timeout=120).strip()
        # Compute the sample ID.
        kid = crypto.sha1_str(sample)
        # Add the new sample to the database.
        db = dbutil.connect(self.db_path)
        c = db.cursor()
        dbutil.sql_insert_dict(c, 'ContentFiles',
                               {'id': kid, 'contents': sample},
                               ignore_existing=True)
        c.close()
        db.commit()
        db.close()
        # Update progress bar.
        progress = self.progress()
        if not logging.get_verbosity() == logging.DEBUG:
          bar.update(progress)
        sample_time = time() - sample_time
        self.sampler.stats['progress'] = progress
        self.sampler.stats['time'] += sample_time
        self.sampler._FlushMeta(self.cache)
        # Determine if we are done sampling.
        if self.term_condition():
          self.producer.stop()
          return
    finally:
      # Always kill the sampler thread.
      print()
      self.producer.stop()
