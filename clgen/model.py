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
CLgen model.
"""
import numpy as np
import os
import re
import sys
import tarfile
import time

from copy import deepcopy
from glob import iglob
from labm8 import fs
from labm8 import system
from labm8 import time as labtime
from six import string_types
from six.moves import cPickle
from tempfile import mktemp

import clgen
from clgen import cache
from clgen import log
from clgen.cache import Cache
from clgen.corpus import Corpus


def get_default_author() -> str:
    """
    Get the default author name for CLgen dist models.

    If CLGEN_AUTHOR environment variable is set, use that. Else, author
    is $USER@$HOSTNAME.

    Returns:
        str: Author name.
    """
    return os.environ.get(
        "CLGEN_AUTHOR",
        "{user}@{host}".format(user=system.USERNAME, host=system.HOSTNAME))


# Default options used for model. Any values provided by the user will override
# these defaults.
DEFAULT_MODEL_OPTS = {
    "author": get_default_author(),
    "architecture": {
      "model_type": "lstm",  # {lstm,rnn.gru}
      "rnn_size": 128,  # num nodes in layer
      "num_layers": 2,  # num layers
    },
    "train_opts": {
      "epochs": 10,
      "grad_clip": 5,
      "learning_rate": 2e-3,  # initial learning rate
      "lr_decary_rate": 5,  # % to reduce learning rate by per epoch
      "intermediate_checkpoints": True
    }
}


class ModelError(clgen.CLgenError):
    """
    Module level error
    """
    pass


class DistError(ModelError):
    """
    Dist model import or export error.
    """
    pass


class Model(clgen.CLgenObject):
    """
    A CLgen Model.
    """
    def __init__(self, corpus: Corpus, **opts):
        """
        Instantiate model.

        Arguments:
            corpus (Corpus): Corpus instance.
            opts (dict): Training options.
        """
        assert(isinstance(corpus, Corpus))

        # Validate options
        for key in opts.keys():
            if key not in DEFAULT_MODEL_OPTS:
                raise clgen.UserError(
                    "Unsupported model option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_MODEL_OPTS.keys()))))

        # set properties
        self.opts = clgen.update(deepcopy(DEFAULT_MODEL_OPTS), opts)
        self.corpus = corpus
        self.hash = self._hash(self.corpus, self.opts)
        self.cache = Cache(fs.path("model", self.hash))

        log.debug("model", self.hash)

    def _hash(self, corpus: Corpus, opts: dict) -> str:
        """ compute model hash """
        hashopts = deepcopy(opts)
        hashopts["train_opts"].pop("epochs")
        return clgen.checksum_list(corpus.hash, *clgen.dict_values(hashopts))

    def _init_tensorflow(self, infer: bool=False):
        """
        Deferred importing of tensorflow and initializing model for training
        or sampling.

        This is necessary for two reasons: first, the tensorflow graph is
        different for training and inference, so must be reset when switching
        between modes. Second, importing tensorflow takes a long time, so
        we only want to do it if we actually need to.

        Arguments:
            infer (bool): If True, initialize model for inference. If False,
                initialize model for training.

        Returns:
            module: imported TensorFlow module
        """
        import tensorflow as tf
        from tensorflow.python.ops import rnn_cell
        from tensorflow.python.ops import seq2seq

        # Use self.tensorflow_state to mark whether or not model is configured
        # for training or inference.
        try:
            if self.tensorflow_state == infer:
                return tf
        except AttributeError:
            pass

        self.cell_fn = {
            "lstm": rnn_cell.BasicLSTMCell,
            "gru": rnn_cell.GRUCell,
            "rnn": rnn_cell.BasicRNNCell
        }.get(self.model_type, None)
        if self.cell_fn is None:
            raise clgen.UserError("Unrecognized model type")

        # reset the graph when switching between training and inference
        tf.reset_default_graph()

        # corpus info:
        batch_size = 1 if infer else self.corpus.batch_size
        seq_length = 1 if infer else self.corpus.seq_length
        vocab_size = self.corpus.vocab_size

        fs.mkdir(self.cache.path)

        cell = self.cell_fn(self.rnn_size, state_is_tuple=True)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * self.num_layers,
                                                 state_is_tuple=True)
        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        scope_name = 'rnnlm'
        with tf.variable_scope(scope_name):
            softmax_w = tf.get_variable("softmax_w",
                                        [self.rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding",
                                            [vocab_size, self.rnn_size])
                inputs = tf.split(
                    1, seq_length, tf.nn.embedding_lookup(
                        embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(
            inputs, self.initial_state, cell,
            loop_function=loop if infer else None, scope=scope_name)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * seq_length])],
            vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # set model status
        self.tensorflow_state = infer

        return tf


    def _get_params_path(self, ckpt):
        """ return path to checkpoint closest to target num of epochs """
        paths = ckpt.all_model_checkpoint_paths
        batch_nums = [int(x.split('-')[-1]) for x in paths]
        epoch_nums = [int((x + 1) / (self.corpus.num_batches))
                      for x in batch_nums]

        closest = self.epochs
        closest_path = None
        for e, path in zip(epoch_nums, paths):
            diff = self.epochs - e
            if diff >= 0 and diff < closest:
                log.verbose("  cached checkpoint at epoch =", e, "diff =", diff)
                closest = diff
                closest_path = path

        return closest_path, paths


    def train(self, quiet=False):
        """
        Train model.
        """
        tf = self._init_tensorflow(infer=False)

        # training options
        learning_rate = self.train_opts["learning_rate"]
        decay_rate = self.train_opts["lr_decary_rate"]
        checkpoint_path = fs.path(self.cache.path, "model.ckpt")

        # resume from prior checkpoint
        ckpt_path, ckpt_paths = None, None
        if self.checkpoint_path:
            # check if all necessary files exist
            assert(fs.isdir(self.checkpoint_path))
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            assert(ckpt)
            assert(ckpt.model_checkpoint_path)
            ckpt_path, ckpt_paths = self._get_params_path(ckpt)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # keep all checkpoints
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # restore model from closest checkpoint
            if ckpt_path:
                log.debug("restoring", ckpt_path)
                saver.restore(sess, ckpt_path)
                log.info("restored checkpoint {}".format(ckpt_path))

            # make sure we don't lose track of other checkpoints
            if ckpt_paths:
                saver.recover_last_checkpoints(ckpt_paths)

            start_batch = sess.run(self.epoch) * self.corpus.num_batches
            batch_count = 0
            total_elapsed = 0
            eta_d, eta_h, eta_m = 0, 0, 0

            for e in range(sess.run(self.epoch) + 1, self.epochs + 1):
                if quiet:
                    log.info("epoch", e, "of", self.epochs + 1)

                # decay and set learning rate
                new_learning_rate = learning_rate * (
                    (float(100 - decay_rate) / 100.0) ** (e - 1))
                sess.run(tf.assign(self.learning_rate, new_learning_rate))
                sess.run(tf.assign(self.epoch, e))

                self.corpus.create_batches()

                state = sess.run(self.initial_state)
                for b in range(self.corpus.num_batches):
                    time_start = time.time()
                    batch_count += 1
                    x, y = self.corpus.next_batch()
                    feed = {self.input_data: x, self.targets: y}
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    train_loss, state, _ = sess.run(
                        [self.cost, self.final_state, self.train_op], feed)
                    batch_num = (e - 1) * self.corpus.num_batches + b
                    max_batch = self.epochs * self.corpus.num_batches

                    progress = float((batch_num + 1 - start_batch) /
                                     (max_batch - start_batch))

                    time_end = time.time()
                    elapsed = time_end - time_start

                    if not quiet:
                        total_elapsed += elapsed
                        avg_elapsed = total_elapsed / batch_count
                        remaining_time = (max_batch - batch_count) * avg_elapsed
                        eta_h, eta_m = divmod(remaining_time / 60, 60)
                        eta_d, eta_h = divmod(eta_h, 24)

                        print(
                            "\r\033[K"
                            "{progress:2.1f}%  {size}x{layers}x{max_epoch} "
                            "{model}  "
                            "batch={batch_num}/{max_batch}  "
                            "epoch={epoch_num}/{max_epoch}  "
                            "lr={lr:.6f}  "
                            "loss={tloss:.3f}  "
                            "time/batch={time_batch:.3f}s  "
                            "eta={eta_d}d{eta_h}h{eta_m:02d}m".format(
                                size=self.rnn_size,
                                layers=self.num_layers,
                                model=self.model_type.upper(),
                                progress=progress * 100,
                                batch_num=b + 1,
                                max_batch=self.corpus.num_batches,
                                epoch_num=e,
                                max_epoch=self.epochs,
                                lr=new_learning_rate,
                                tloss=train_loss,
                                time_batch=avg_elapsed,
                                eta_d=int(eta_d),
                                eta_h=int(eta_h),
                                eta_m=int(eta_m)), end="")

                save = self.opts["train_opts"]["intermediate_checkpoints"]
                save |= e == self.epochs  # last epoch
                if save:
                    if not quiet:
                        print()
                    saver.save(sess, checkpoint_path, global_step=batch_num)
                    log.info("model saved to {}".format(checkpoint_path))

    def sample(self, seed_text="__kernel void", output=sys.stdout,
               num_samples=1, temperature=1, max_length=10000, seed=None,
               quiet=False):
        """
        Sample model.

        Arguments:
            seed_text (str, optional): Sample start text
            output (file handler, optional): Where to print output to
            num_samples (int, optional): Number of samples to generated
            temperature (float, optional): Sampling temperature
            max_length (int, optional): Maximum sample length
            seed (int, optional): If set, set random number seed for
                reproducible samples. If None, set no seed.
            quiet (bool, optional): If False, stream output to stdout
        """
        tf = self._init_tensorflow(infer=True)

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.cache.path)

            assert(ckpt)
            assert(ckpt.model_checkpoint_path)

            saver.restore(sess, ckpt.model_checkpoint_path)

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return int(np.searchsorted(t, np.random.rand(1) * s))

            start_depth = 0
            start_started = False
            for char in seed_text:
                if char == '{':
                    start_depth += 1
                    start_started = True
                elif char == '}':
                    start_depth -= 1

            for i in range(1, num_samples + 1):
                state = sess.run(self.cell.zero_state(1, tf.float32))
                depth = start_depth  # function block depth
                started = start_started

                seed_tensor = self.corpus.atomizer.atomize(seed_text)
                for index in seed_tensor[:-1]:
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {self.input_data: x, self.initial_state: state}
                    [state] = sess.run([self.final_state], feed)

                sampling_type = 1  # default
                output.write("\n\n/* SAMPLE {} */\n\n".format(i))
                output.write(seed_text)
                if not quiet:
                    sys.stdout.write("\n\n/* SAMPLE {} */\n\n".format(i))
                    sys.stdout.write(seed_text)
                    sys.stdout.flush()

                ret = seed_text
                index = seed_tensor[-1]

                for _ in range(max_length):
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {self.input_data: x, self.initial_state: state}
                    [probs, state] = sess.run(
                        [self.probs, self.final_state], feed)
                    p = probs[0]

                    if sampling_type == 0:
                        # use max at each step
                        sample = np.argmax(p)
                    elif sampling_type == 2:
                        # sample on space
                        if atom == ' ':
                            sample = weighted_pick(p)
                        else:
                            sample = np.argmax(p)
                    else:
                        # sample on each step
                        sample = weighted_pick(p)

                    index = sample
                    atom = self.corpus.atomizer.deatomize([sample])
                    ret += atom

                    output.write(atom)
                    if not quiet:
                        sys.stdout.write(atom)

                    # update function block depth
                    for char in atom:
                        if char == '{':
                            started = True
                            depth += 1
                        elif char == '}':
                            depth -= 1
                    # stop sampling if depth = 0
                    if started and depth == 0:
                        break

            if not quiet:
                sys.stdout.write('\n\n')

    @property
    def model_type(self) -> str:
        return self.opts["architecture"]["model_type"]

    @property
    def rnn_size(self) -> int:
        return self.opts["architecture"]["rnn_size"]

    @property
    def num_layers(self) -> int:
        return self.opts["architecture"]["num_layers"]

    @property
    def grad_clip(self) -> int:
        return self.train_opts["grad_clip"]

    @property
    def epochs(self) -> int:
        return self.train_opts["epochs"]

    @property
    def train_opts(self) -> dict:
        return self.opts["train_opts"]

    @property
    def meta(self):
        """
        Get trained model metadata.

        Format spec: https://github.com/ChrisCummins/clgen/issues/25

        Returns:
            dict: Metadata.
        """
        # checksum corpus and model cache files. Paths are relative to cache
        # root.
        cache_root_re = r'^' + cache.ROOT + '/'
        corpus_files = dict(
            (re.sub(cache_root_re, "", x), clgen.checksum_file(x))
            for x in fs.ls(self.corpus.cache.path, abspaths=True))
        model_files = dict(
            (re.sub(cache_root_re, "", x), clgen.checksum_file(x))
            for x in fs.ls(self.cache.path, abspaths=True))

        contents = corpus_files.copy()
        contents.update(model_files)

        _meta = deepcopy(self.opts)
        _meta["version"] = clgen.version()
        _meta["date_packaged"] = labtime.nowstr()
        _meta["corpus"] = self.corpus.meta,
        _meta["contents"] = contents

        return _meta

    def to_dist(self, distpath, author=None):
        """
        Create a dist file.

        Arguments:
            distpath (str): Path to dist file.
            author (str, optional): Author name.

        Returns:
            str: Path to generated distfile.
        """
        outpath = fs.abspath(distpath) + ".tar.bz2"
        if fs.exists(outpath):
            raise DistError("file {} exists".format(outpath))

        meta = self.meta
        if author is not None:
            meta["author"] = author
        log.debug(clgen.format_json(meta))

        try:
            tar = tarfile.open(outpath, 'w:bz2')

            # write meta
            metapath = mktemp(prefix="clgen-", suffix=".json")
            clgen.write_file(metapath, clgen.format_json(meta))
            log.debug("metafile:", metapath)

            # create tarball
            tar.add(metapath, arcname="meta.json")

            # pack contents:
            for path in meta["contents"]:
                abspath = fs.path(cache.ROOT, path)
                log.verbose("packing", abspath)
                tar.add(abspath, arcname=fs.path("contents", path))

            # tidy up
            fs.rm(metapath)
            tar.close()
        except Exception as e:
            tar.close()
            fs.rm(metapath)
            fs.rm(outpath)
            raise e

        return outpath

    def __repr__(self):
        """
        String representation.
        """
        return "{hash}: {data}".format(
            hash=self.hash, data=clgen.format_json(self.opts))

    @property
    def checkpoint_path(self):
        """
        Get path to most checkpoint, if exists.

        Returns:

            str or None: Path to checkpoint, or None if no checkpoints.
        """
        if self.cache["checkpoint"]:
            return self.cache.path
        else:
            return None


class DistModel(Model):
    """
    Distributed model.

    A distmodel is a pre-trained model, distributed without training corpus.
    """
    def __init__(self, tarpath):
        """
        Instantiate distmodel.

        Arguments:
            tarpath (str): Path to distmodel.
        """
        assert(isinstance(tarpath, string_types))

        disthash = clgen.checksum_file(tarpath)
        self.cache = Cache(fs.path("dist", disthash))

        # unpack archive if necessary
        unpacked = False
        if not self.cache['meta.json']:
            log.info("unpacking distmodel", tarpath)
            clgen.unpack_archive(tarpath, dir=self.cache.path)
            unpacked = True
            if not self.cache['meta.json']:
                raise DistError("meta.json not in '{}'".format(tarpath))

        self._meta = clgen.load_json_file(self.cache['meta.json'])
        if unpacked:
            for path in self._meta['contents']:
                fs.mkdir(fs.path(cache.ROOT, fs.dirname(path)))
                cache_path = fs.path(self.cache['contents'], path)
                dest_path = fs.path(cache.ROOT, path)
                log.verbose("unpacking", path)
                os.rename(cache_path, dest_path)

            self.validate()

        # FIXME:
        self.opts = self.meta

        log.info("distfile model: ", disthash)
        if "author" in self.meta:
            log.info("distfile author:", self.meta["author"])
        if "date packaged" in self.meta:
            log.info("distfile date:  ", self.meta["date_packaged"])

        # Invoke superconstructor, to proceed as a normal model.
        # FIXME:
        super(DistModel, self).__init__(
            Corpus.from_json(self.meta["corpus"]), self.meta["train_opts"])

    @property
    def meta(self):
        """
        Get model metadata.

        Format spec: https://github.com/ChrisCummins/clgen/issues/25

        Returns:
            dict: Metadata.
        """
        return self._meta

    def validate(self):
        """
        Validate contents of a distfile.

        Returns:
            bool: True.

        Raises:
            DistError: In case of invalid distfile.
        """
        version = self.meta.get("version", None)
        version_1_re = r'0\.1\.[0-9]+'

        # before version 0.1.1, distfiles did not contain version string
        if version is None:
            if not re.match(version_1_re, clgen.version()):
                log.fatal("distfile is incompatible with CLgen version {}. "
                          "Please install CLgen 0.1.7."
                          .format(clgen.version()))

            if not len(fs.ls(self.cache.path)) == 3:
                log.error("Unpackaed tar contents:\n  ",
                          "\n  ".join(fs.ls(self.cache.path, abspaths=True)))
                raise DistError("Bad distfile")
            if not self.cache['model.json']:
                raise DistError("model.json not in disfile")
            if not self.cache['checkpoint']:
                raise DistError("checkpoint not in disfile")
            return True

        contents = self.meta["contents"]

        # versions 0.1.1 - 0.1.7:
        if re.match(version_1_re, version):
            if not re.match(version_1_re, clgen.version()):
                log.fatal("distfile version {} is incompatible with CLgen "
                          "version {}. Please install CLgen 0.1.7."
                          .format(version, clgen.version()))

            if version != clgen.version():
                log.warning("distfile version {} does not match CLgen version "
                            "{}. There may be incompabilities"
                            .format(version, clgen.version()))
            for file in contents:
                # compare unpacked file contents to expected checksums
                path = self.cache[file]
                checksum = clgen.checksum_file(path)
                log.verbose("  expected checksum:", file, contents[file])
                log.verbose("calculated checksum:", file, checksum)
                log.verbose()
                if checksum != contents[file]:
                    raise DistError(
                        "distfile {} checksum failed".format(file))
            return True

        # version 0.2.x:
        for path in contents:
            # compare unpacked file contents to expected checksums
            abspath = fs.path(cache.ROOT, path)
            checksum = clgen.checksum_file(abspath)
            log.verbose("  expected checksum:", path, contents[path])
            log.verbose("calculated checksum:", path, checksum)
            if checksum != contents[path]:
                raise DistError(
                    "distfile {} checksum failed".format(path))
        return True


def from_json(model_json: dict) -> Model:
    """
    Load model from JSON.

    Arguments:
        model_json (dict): JSON specification.

    Returns:
        Model: Model instance.
    """
    assert(type(model_json) is dict)

    if "corpus" not in model_json:
        raise clgen.UserError("model JSON has no corpus entry")

    # create corpus and remove from JSON
    corpus = Corpus.from_json(model_json.pop("corpus"))

    return Model(corpus, **model_json)


def from_tar(path):
    """
    Load model from tarball.

    Arguments:
        path (str): Path to tarball.

    Returns:
        DistModel: Model instance.

    Raises:
        File404: If path does not exist.
        DistError: If distfile is malformed.
    """
    assert(isinstance(path, string_types))

    path = fs.path(path)
    if not fs.isfile(path):
        raise clgen.File404("distfile not found '{}'".format(path))

    return DistModel(path)
