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
CLgen model.
"""
from __future__ import print_function

import json
import numpy as np
import os
import re
import sys
import tarfile
import tensorflow as tf
import time

from copy import copy
from glob import glob, iglob
from labm8 import fs
from labm8 import system
from labm8 import time as labtime
from six import string_types
from six.moves import cPickle
from tempfile import mktemp
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import clgen
from clgen import log
from clgen.cache import Cache
from clgen.corpus import Corpus


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


def get_default_author():
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


class Model(clgen.CLgenObject):
    """
    A CLgen Model.
    """
    def __init__(self, corpus, train_opts, infer=False):
        """
        Instantiate model.

        Arguments:
            corpus (Corpus): Corpus instance.
            train_opts (dict): Training options.
        """
        assert(isinstance(corpus, Corpus))
        assert(type(train_opts) is dict)

        self.corpus = corpus
        self.train_opts = train_opts

        self.hash = self._hash(train_opts, corpus)
        self.cache = Cache(fs.path("model", self.hash))

        # Parse model options:
        # TODO: Change model format so that model type, rnn size, and num layers
        # are in the top level specification, not in "train_opts" dict.
        self.model_type = self.train_opts.get("model_type", "lstm")
        self.rnn_size = self.train_opts.get("rnn_size", 128)
        self.num_layers = self.train_opts.get("num_layers", 2)
        self.grad_clip = self.train_opts.get("grad_clip", 5)

        self.cell_fn = {
            "lstm": rnn_cell.BasicLSTMCell,
            "gru": rnn_cell.GRUCell,
            "rnn": rnn_cell.BasicRNNCell
        }.get(self.model_type, None)
        if self.cell_fn is None:
            raise clgen.UserError("Unrecognized model type")

    def _hash(self, train_opts, corpus):
        checksum_data = sorted(
            [str(x) for x in train_opts.values()] +
            [corpus.hash])
        string = "".join(checksum_data)
        return clgen.checksum_str(string)

    def _init_tensorflow(self, infer=False):
        try:
            if self.tensorflow_state == infer:
                return
        except AttributeError:
            pass

        tf.reset_default_graph()

        # Corpus info:
        batch_size = 1 if infer else self.corpus.batch_size
        seq_length = 1 if infer else self.corpus.seq_length
        vocab_size = self.corpus.vocab_size

        tmp_chars_vocab_path = fs.path(self.cache.path, "chars_vocab.tmp.pkl")
        with open(tmp_chars_vocab_path, 'wb') as outfile:
            cPickle.dump((self.corpus.chars, self.corpus.vocab), outfile)
        self.cache["chars_vocab.pkl"] = tmp_chars_vocab_path

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
            # TODO: Determine device
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


    def train(self):
        """
        Train model.
        """
        self._init_tensorflow(infer=False)

        max_epochs = self.train_opts.get("max_epochs", 50)
        learning_rate = self.train_opts.get("learning_rate", 2e-3)
        decay_rate = self.train_opts.get("lr_decary_rate", 5)
        checkpoint_path = fs.path(self.cache.path, "model.ckpt")

        # resume from prior checkpoint
        if self.most_recent_checkpoint:
            # open saved vocab/dict and check if vocabs/dicts are compatible
            assert(fs.isfile(self.cache["chars_vocab.pkl"]))
            # FIXME:
            # with open(self.cache["chars_vocab.pkl"]) as infile:
            #     saved_chars, saved_vocab = cPickle.load(infile)
            # assert(saved_chars == self.corpus.chars)
            # assert(saved_vocab == self.corpus.vocab)

            # check if all necessary files exist
            assert(fs.isdir(self.most_recent_checkpoint))
            ckpt = tf.train.get_checkpoint_state(self.most_recent_checkpoint)
            assert(ckpt)
            assert(ckpt.model_checkpoint_path)
            log.debug("loaded checkpoint {}".format(ckpt.model_checkpoint_path))

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())

            # restore model from checkpoint
            if self.most_recent_checkpoint:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for e in range(sess.run(self.epoch), max_epochs):
                # decay and set learning rate
                sess.run(tf.assign(self.learning_rate, learning_rate * (decay_rate ** e)))
                sess.run(tf.assign(self.epoch, e))

                self.corpus.reset_batch_pointer()
                state = sess.run(self.initial_state)
                for b in range(self.corpus.num_batches):
                    time_start = time.time()
                    x, y = self.corpus.next_batch()
                    feed = {self.input_data: x, self.targets: y}
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    train_loss, state, _ = sess.run(
                        [self.cost, self.final_state, self.train_op], feed)
                    time_end = time.time()
                    batch_num = e * self.corpus.num_batches + b
                    max_batch = max_epochs * self.corpus.num_batches
                    print("{}/{} (epoch {}/{}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(batch_num + 1, max_batch,
                                  e, max_epochs,
                                  train_loss, time_end - time_start))
                    # save_checkpoint = batch_num % checkpoint_every == 0
                    # save_checkpoint |= (e == max_epochs - 1
                    #                     and b == self.corpus.num_batches - 1)
                    # if save_checkpoint:
                saver.save(sess, checkpoint_path, global_step=batch_num)
                log.info("model saved to {}".format(checkpoint_path))

    def sample(self, seed_text, output=sys.stdout, num_samples=1,
               temperature=.75, max_length=10000, quiet=False):
        """
        Sample model.
        """
        self._init_tensorflow(infer=True)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(self.cache.path)

            assert(ckpt)
            assert(ckpt.model_checkpoint_path)

            with open(self.cache["chars_vocab.pkl"], "rb") as infile:
                chars, vocab = cPickle.load(infile)

            saver.restore(sess, ckpt.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            depth = 0  # function block depth
            started = False

            for char in seed_text[:-1]:
                if char == '{':
                    depth += 1
                    started = True
                elif char == '}':
                    depth -= 1

                x = np.zeros((1, 1))
                x[0, 0] = vocab[char]
                feed = {self.input_data: x, self.initial_state: state}
                [state] = sess.run([self.final_state], feed)

            output.write(seed_text)
            if not quiet:
                sys.stdout.write(seed_text)

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return int(np.searchsorted(t, np.random.rand(1) * s))

            sampling_type = 1  # default

            ret = seed_text
            char = seed_text[-1]

            for n in range(max_length):
                x = np.zeros((1, 1))

                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if char == ' ':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = chars[sample]
                ret += pred
                char = pred

                output.write(pred)
                if not quiet:
                    sys.stdout.write(pred)

                # update function block depth
                if char == '{':
                    started = True
                    depth += 1
                elif char == '}':
                    depth -= 1
                # stop sampling if depth = 0
                if started and depth == 0:
                    break

    @property
    def meta(self):
        """
        Get trained model metadata.

        Format spec: https://github.com/ChrisCummins/clgen/issues/25

        Returns:
            dict: Metadata.

        Raises:
            DistError: If model has not been trained.
        """
        dist_t7 = self.most_recent_checkpoint

        if dist_t7 is None:
            raise DistError("model is untrained")

        dist_json = re.sub(r'.t7$', '.json', dist_t7)

        if not fs.exists(dist_json):  # sanity check
            raise clgen.InternalError(
                "Checkpoint {t7} does not have corresponding json file {json}"
                .format(t7=dist_t7, json=dist_json))

        contents = {
            "model.t7": clgen.checksum_file(dist_t7),
            "model.json": clgen.checksum_file(dist_json)
        }

        return {
            "version": clgen.version(),
            "author": get_default_author(),
            "date packaged": labtime.nowstr(),
            "train_opts": self.train_opts,
            "contents": contents
        }

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
            clgen.write_file(metapath, json.dumps(meta))
            log.debug("metafile:", metapath)

            dist_t7 = self.most_recent_checkpoint
            dist_json = re.sub(r'.t7$', '.json', dist_t7)

            # create tarball
            tar.add(metapath, arcname="meta.json")
            tar.add(dist_t7, arcname="model.t7")
            tar.add(dist_json, arcname="model.json")

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
            hash=self.hash, data=clgen.format_json(self.train_opts))

    @property
    def checkpoints(self):
        """
        Training checkpoints.

        Returns:

            str[]: List of paths to checkpoint files.
        """
        return []  # TODO

    @property
    def most_recent_checkpoint(self):
        """
        Get path to most recently created t7 checkpoint.

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

        self.hash = self._hash(tarpath)
        self.cache = Cache(fs.path("model", self.hash))

        # unpack archive if necessary
        unpacked = False
        if not self.cache['model.json'] or not self.cache['model.t7']:
            unpacked = True
            log.info("unpacking", tarpath)
            clgen.unpack_archive(tarpath, dir=self.cache.path)
            if not self.cache['meta.json']:
                raise DistError("meta.json not in '{}'".format(tarpath))

        self._meta = clgen.load_json_file(self.cache['meta.json'])
        if unpacked:
            self.validate()

        self.train_opts = self.meta['train_opts']

        log.info("distfile model: ", self.hash)
        if "author" in self.meta:
            log.verbose("distfile author:", self.meta["author"])
        if "date packaged" in self.meta:
            log.verbose("distfile date:  ", self.meta["date packaged"])

    def _hash(self, tarpath):
        return clgen.checksum_file(tarpath)

    @property
    def meta(self):
        """
        Get model metadata.

        Format spec: https://github.com/ChrisCummins/clgen/issues/25

        Returns:
            dict: Metadata.
        """
        return self._meta

    def train(self):
        """
        This method does nothing, distmodels are pre-tained.
        """
        pass

    @property
    def checkpoints(self):
        """
        Training checkpoints.

        Returns:

            str[]: List of paths to checkpoint files.
        """
        return [self.most_recent_checkpoint()]

    @property
    def most_recent_checkpoint(self):
        """
        Get path to pre-trained t7 checkpoint.

        Returns:

            str or None: Path to checkpoint.
        """
        return self.cache['model.t7']

    def validate(self):
        """
        Validate contents of a distfile.

        Returns:
            bool: True.

        Raises:
            DistError: In case of invalid distfile.
        """
        version = self.meta.get("version", None)

        if version is None:
            # before version 0.1.1, distfiles did not contain version string
            if not len(fs.ls(self.cache.path)) == 3:
                log.error("Unpackaed tar contents:\n  ",
                          "\n  ".join(fs.ls(self.cache.path, abspaths=True)))
                raise DistError("Bad distfile")
            if not self.cache['model.json']:
                raise DistError("model.json not in disfile")
            if not self.cache['model.t7']:
                raise DistError("model.t7 not in disfile")
            return True

        if version != clgen.version():
            log.warning("distfile version {} does not match CLgen version "
                        "{}. There may be incompabilities"
                        .format(version, clgen.version()))
        contents = self.meta["contents"]
        for file in contents:
            # compare unpacked file contents to expected checksums
            path = self.cache[file]
            checksum = clgen.checksum_file(path)
            log.verbose("  expected checksum:", file, contents[file])
            log.verbose("calculated checksum:", file, checksum)
            if checksum != contents[file]:
                raise DistError(
                    "distfile {} checksum failed".format(file))

        return True


def from_json(model_json):
    """
    Load model from JSON.

    Arguments:
        model_json (dict): JSON specification.

    Returns:
        Model: Model instance.
    """
    assert(type(model_json) is dict)

    corpus = Corpus.from_json(model_json["corpus"])
    train_opts = model_json["train_opts"]

    # validate train_opts flags
    valid_opts = [
        "batch_size", "seq_length", "model_type", "rnn_size", "num_layers",
        "dropout", "batchnorm", "learning_rate", "max_epochs", "grad_clip",
        "lr_decay_every", "lr_decay_factor",
    ]
    for key in train_opts.keys():
        if key not in valid_opts:
            raise clgen.UserError(
                "Unrecognized training option '{}'".format(key))

    # FIXME: set infer
    return Model(corpus, train_opts, infer=True)


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
