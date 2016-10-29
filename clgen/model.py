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

import os

from glob import glob, iglob
from labm8 import fs

import clgen
from clgen import log
from clgen import torch_rnn
from clgen.cache import Cache
from clgen.corpus import Corpus


class Model(clgen.CLgenObject):
    def __init__(self, corpus, train_opts):
        assert(isinstance(corpus, Corpus))
        assert(type(train_opts) is dict)

        self.hash = corpus.hash
        self.corpus = corpus
        self.train_opts = train_opts

        self.checkpoint_cache = Cache(fs.path(self.hash, "cv"))

    def train(self):
        print("CACHE:", self.checkpoint_cache.path)
        print("CHECKPOINTS:", self.checkpoints)
        print("MOST RECENT:", self.most_recent_checkpoint)

        self.train_opts["checkpoint_every"] = 100
        self.train_opts["checkpoint_name"] = (
            self.checkpoint_cache.path + os.pathsep)

        torch_rnn.train(**self.train_opts)

    @property
    def checkpoints(self):
        return glob(fs.path(self.checkpoint_cache.path, '*.t7'))

    @property
    def most_recent_checkpoint(self):
        return max(iglob(fs.path(self.checkpoint_cache.path, '*.t7')),
                   key=os.path.getctime)


def from_json(model_json):
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
