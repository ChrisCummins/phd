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
import os
import re
import tarfile

from copy import copy
from glob import glob, iglob
from labm8 import fs
from labm8 import system
from labm8 import time
from tempfile import mktemp

import clgen
from clgen import log
from clgen import torch_rnn
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
    return "{user}@{host}".format(user=system.USERNAME, host=system.HOSTNAME)


class Model(clgen.CLgenObject):
    def __init__(self, corpus, train_opts):
        assert(isinstance(corpus, Corpus))
        assert(type(train_opts) is dict)

        self.corpus = corpus
        self.train_opts = train_opts

        self.hash = self._hash(train_opts, corpus)
        self.cache = Cache(fs.path("model", self.hash))

    def _hash(self, train_opts, corpus):
        checksum_data = sorted(
            [str(x) for x in train_opts.values()] +
            [corpus.hash])
        string = "".join([str(x) for x in checksum_data])
        return clgen.checksum_str(string)

    def train(self):
        # assemble training options
        opts = copy(self.train_opts)

        opts["reset_iterations"] = 0
        opts["input_json"] = self.corpus.input_json
        opts["input_h5"] = self.corpus.input_h5
        opts["checkpoint_name"] = fs.path(self.cache.path, "model")

        # set default arguments
        if opts.get("print_every", None) is None:
            opts["print_every"] = 10
        if opts.get("checkpoint_every", None) is None:
            opts["checkpoint_every"] = 100

        # resume from prior checkpoint
        if self.most_recent_checkpoint:
            opts["init_from"] = self.most_recent_checkpoint

        torch_rnn.train(**opts)

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
            "date packaged": time.nowstr(),
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
        return "{hash}: {data}".format(
            hash=self.hash, data=clgen.format_json(self.train_opts))

    @property
    def checkpoints(self):
        return glob(fs.path(self.cache.path, '*.t7'))

    @property
    def most_recent_checkpoint(self):
        """
        Get path to most recently created t7 checkpoint

        Returns:

            str or None: Path to checkpoint, or None if no checkpoints.
        """
        # if there's nothing trained, then max() will raise ValueError because
        # of an empty sequence
        def get_checkpoint_iterations(path):
            return int(re.search("model_([0-9]+)\.t7", path).group(1))

        try:
            return max(iglob(fs.path(self.cache.path, '*.t7')),
                       key=get_checkpoint_iterations)
        except ValueError:
            return None


class DistModel(Model):
    def __init__(self, tarpath):
        assert(type(tarpath) is str)

        self.hash = self._hash(tarpath)
        self.cache = Cache(fs.path("model", self.hash))

        log.info("dist model", self.hash)

        # unpack archive if necessary
        if not self.cache['model.json'] or not self.cache['model.t7']:
            log.info("unpacking", tarpath)
            clgen.unpack_archive(tarpath, dir=self.cache.path)
            if not len(fs.ls(self.cache.path)) == 3:
                log.error("Unpackaed tar contents:\n  ",
                          "\n  ".join(fs.ls(self.cache.path, abspaths=True)))
                raise DistError("Bad distfile '{}'".format(tarpath))
            if not self.cache['model.json']:
                raise DistError("model.json not in '{}'".format(tarpath))
            if not self.cache['model.t7']:
                raise DistError("model.t7 not in '{}'".format(tarpath))
            if not self.cache['meta.json']:
                raise DistError("meta.json not in '{}'".format(tarpath))

        metadata = clgen.load_json_file(self.cache['meta.json'])
        self.train_opts = metadata['train_opts']

    def _hash(self, tarpath):
        return clgen.checksum_file(tarpath)

    def train(self):
        pass

    @property
    def checkpoints(self):
        return [self.most_recent_checkpoint()]

    @property
    def most_recent_checkpoint(self):
        """
        Get path to most recently initialized t7
        """
        return self.cache['model.t7']


def from_json(model_json):
    assert(type(model_json) is dict)

    corpus = Corpus.from_json(model_json["corpus"])
    train_opts = model_json["train_opts"]

    # validate train_opts flags against those accpted by torch-rnn/train.lua
    valid_opts = [
        "batch_size", "seq_length", "model_type", "rnn_size", "num_layers",
        "dropout", "batchnorm", "learning_rate", "max_epochs", "grad_clip",
        "lr_decay_every", "lr_decay_factor",
    ]
    for key in train_opts.keys():
        if key not in valid_opts:
            raise clgen.UserError(
                "Unrecognized training option '{}'".format(key))

    return Model(corpus, train_opts)


def from_tar(path):
    assert(type(path) is str)

    return DistModel(path)
