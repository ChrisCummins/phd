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
Python wrapper around torch-rnn.
"""
import labm8
import sys

from labm8 import fs
from six import iteritems
from subprocess import Popen, PIPE, STDOUT

import clgen
from clgen import log
from clgen import native


class TorchRnnError(clgen.CLgenError):
    pass


class PreprocessError(TorchRnnError):
    pass


class TrainError(TorchRnnError):
    pass


def preprocess(input_txt, output_json, output_h5):
    """
    Wrapper around preprocess script.
    """
    cmd = [
        sys.executable, native.TORCH_RNN_PREPROCESS, '--input_txt', input_txt,
        '--output_json', output_json,
        '--output_h5', output_h5,
    ]

    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise PreprocessError(stderr.decode('utf-8'))

    output = stdout.decode('utf-8')
    log.info(output)


def train(**train_opts):
    """
    Wrapper around torch-rnn train script.

    Arguments:
        **train_opts (dict): Key value options for flags.
    """
    # change to torch-rnn directory
    fs.cd(native.TORCH_RNN_DIR)

    flags = labm8.flatten(
        [('-' + key, str(value)) for key, value in iteritems(train_opts)])
    cmd = [native.TH, "train.lua"] + flags

    log.debug(' '.join([str(x) for x in cmd]))
    process = Popen(cmd)
    process.communicate()

    if process.returncode != 0:
        raise TrainError('torch-rnn training failed with status ' +
                         str(process.returncode))

    # return to previous working directory
    fs.cdpop()


def sample(output, **sample_opts):
    """
    Wrapper around torch-rnn sample script.

    Arguments:
        output (str): Path to output file.
        **train_opts (dict): Key value options for flags.
    """
    # change to torch-rnn directory
    fs.cd(native.TORCH_RNN_DIR)

    flags = labm8.flatten(
        [('-' + key, str(value)) for key, value in iteritems(sample_opts)])
    cmd = [native.TH, "sample.lua"] + flags

    log.debug(' '.join([str(x) for x in cmd]))
    process = Popen(cmd)
    process.communicate()

    if process.returncode != 0:
        raise TrainError('torch-rnn sampling failed with status ' +
                         str(process.returncode))

    # return to previous working directory
    fs.cdpop()
