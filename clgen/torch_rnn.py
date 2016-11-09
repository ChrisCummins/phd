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
from clgen import config as cfg
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


def get_device_flags():
    # TODO: This should be runtime configurable
    gpu_id = 0

    if cfg.USE_CUDA:
        return {"gpu": gpu_id}
    # TODO: it apperas that cltorch can no longer be installed using this
    # method, but instead requires using a fork of the torch distro with
    # OpenCL support baked in. It would be fairly substantial job to add
    # support for this second torch distro, so I'm going to ignore it and
    # simply disable OpenCL support for torch. See:
    #   https://github.com/hughperkins/distro-cl
    #
    # elif cfg.USE_OPENCL:
    #     return {"gpu": gpu_id, "gpu_backend": "opencl"}
    else:  # cpu-only
        return {"gpu": -1}


def train(**train_opts):
    """
    Wrapper around torch-rnn train script.

    Arguments:
        **train_opts (dict): Key value options for flags.
    """
    # change to torch-rnn directory
    fs.cd(native.TORCH_RNN_DIR)

    train_opts.update(get_device_flags())

    flags = labm8.flatten(
        [('-' + key, str(value)) for key, value in iteritems(train_opts)])
    cmd = [native.TH, "train.lua"] + flags

    log.debug('(cd {dir} && {cmd})'.format(
        dir=native.TORCH_RNN_DIR,
        cmd=' '.join([str(x) for x in cmd])))

    process = Popen(cmd, stdout=PIPE)
    while True:  # print stdout in real-time
        line = process.stdout.readline()
        sys.stdout.write(line.decode('utf-8'))
        if not line:
            break

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

    sample_opts.update(get_device_flags())

    flags = labm8.flatten(
        [('-' + key, str(value)) for key, value in iteritems(sample_opts)])
    cmd = [native.TH, "sample.lua"] + flags

    log.debug('(cd {dir} && {cmd})'.format(
        dir=native.TORCH_RNN_DIR,
        cmd=' '.join([str(x) for x in cmd])))

    process = Popen(cmd, stdout=PIPE)
    # TODO: Consider optimizing using a second tee process and pipe
    with open(output, "wb") as outfile:
        for line in process.stdout:
            sys.stdout.write(line.decode('utf-8'))
            outfile.write(line)
    process.wait()

    if process.returncode != 0:
        raise TrainError('torch-rnn sampling failed with status ' +
                         str(process.returncode))

    # return to previous working directory
    fs.cdpop()
