#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
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
import pytest
from clgen import test as tests

import os

from clgen import cli
from labm8 import fs


def _mymethod(a, b):
    c = a // b
    print("{a} / {b} = {c}".format(**vars()))
    return c


def test_run():
    assert cli.run(_mymethod, 4, 2) == 2


def test_run_exception_handler():
    os.environ["DEBUG"] = ""
    with pytest.raises(SystemExit):
        cli.run(_mymethod, 1, 0)


def test_run_exception_debug():
    os.environ["DEBUG"] = "1"
    with pytest.raises(ZeroDivisionError):
        cli.run(_mymethod, 1, 0)


def test_cli_version():
    with pytest.raises(SystemExit):
        cli.main("--version")


def test_cli_test_cache_path():
    with pytest.raises(SystemExit):
        cli.main("test --cache-path".split())


def test_cli_test_coverage_path():
    with pytest.raises(SystemExit):
        cli.main("test --coverage-path".split())


def test_cli_test_coveragerc_path():
    with pytest.raises(SystemExit):
        cli.main("test --coveragerc-path".split())


def test_cli():
    fs.rm("kernels.db")
    cli.main("db init kernels.db".split())
    assert fs.exists("kernels.db")

    corpus_path = tests.archive("tiny", "corpus")
    cli.main("db explore kernels.db".split())
    cli.main(f"fetch fs kernels.db {corpus_path}".split())
    cli.main("preprocess kernels.db".split())
    cli.main("db explore kernels.db".split())

    fs.rm("kernels_out")
    cli.main("db dump kernels.db -d kernels_out".split())
    assert fs.isdir("kernels_out")
    assert len(fs.ls("kernels_out")) >= 1

    fs.rm("kernels.cl")
    cli.main("db dump kernels.db kernels.cl --file-sep --eof --reverse".split())
    assert fs.isfile("kernels.cl")

    fs.rm("kernels_out")
    cli.main("db dump kernels.db --input-samples -d kernels_out".split())
    assert fs.isdir("kernels_out")
    assert len(fs.ls("kernels_out")) == 250

    fs.rm("kernels.db")
    fs.rm("kernels_out")


def test_cli_train():
    with tests.chdir(tests.data_path("pico")):
        cli.main("train model.json".split())
        cli.main("--corpus-dir model.json".split())
        cli.main("--model-dir model.json".split())


def test_cli_sample():
    with tests.chdir(tests.data_path("pico")):
        cli.main("sample model.json sampler.json".split())
        cli.main("--corpus-dir model.json".split())
        cli.main("--model-dir model.json".split())
        cli.main("--sampler-dir model.json sampler.json".split())
        cli.main("ls files model.json sampler.json".split())


def test_cli_ls():
    cli.main("ls models".split())
    cli.main("ls samplers".split())
