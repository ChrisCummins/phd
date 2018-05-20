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

import deeplearning.clgen.clgen.errors
import deeplearning.clgen.clgen.package_util
from deeplearning.clgen import clgen


def test_pacakge_data():
  with pytest.raises(deeplearning.clgen.clgen.errors.InternalError):
    deeplearning.clgen.clgen.package_util.package_data(
      "This definitely isn't a real path")
  with pytest.raises(deeplearning.clgen.clgen.errors.File404):
    deeplearning.clgen.clgen.package_util.package_data(
      "This definitely isn't a real path")


def test_pacakge_str():
  with pytest.raises(deeplearning.clgen.clgen.errors.InternalError):
    deeplearning.clgen.clgen.package_util.package_str(
      "This definitely isn't a real path")
  with pytest.raises(deeplearning.clgen.clgen.errors.File404):
    deeplearning.clgen.clgen.package_util.package_str(
      "This definitely isn't a real path")


def test_sql_script():
  with pytest.raises(deeplearning.clgen.clgen.errors.InternalError):
    deeplearning.clgen.clgen.package_util.sql_script(
      "This definitely isn't a real path")
  with pytest.raises(deeplearning.clgen.clgen.errors.File404):
    deeplearning.clgen.clgen.package_util.sql_script(
      "This definitely isn't a real path")


def test_platform_info():
  clgen.platform_info()
