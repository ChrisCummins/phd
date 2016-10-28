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
import sys

from subprocess import Popen, PIPE, STDOUT

from clgen import log
from clgen import native


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
        raise PreProcessError(stderr.decode('utf-8'))

    output = stdout.decode('utf-8')
    log.info(output)
