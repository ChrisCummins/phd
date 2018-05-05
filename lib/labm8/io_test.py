# Copyright (C) 2015-2018 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
import re

from lib import labm8 as lab
from lib.labm8 import io
from lib.labm8.tests.testutil import TestCase

if lab.is_python3():
  from io import StringIO
else:
  from StringIO import StringIO


class TestIO(TestCase):

  # colourise()
  def test_colourise(self):
    self._test("\033[91mHello, World!\033[0m",
               io.colourise(io.Colours.RED, "Hello, World!"))

  # printf()
  def test_printf(self):
    out = StringIO()
    io.printf(io.Colours.RED, "Hello, World!", file=out)
    self._test("\033[91mHello, World!\033[0m", out.getvalue().strip())

  # pprint()
  def test_pprint(self):
    out = StringIO()
    io.pprint({"foo": 1, "bar": "baz"}, file=out)
    self._test('{\n  "bar": "baz",\n  "foo": 1\n}', out.getvalue().strip())

  # info()
  def test_info(self):
    out = StringIO()
    io.info("foo", file=out)
    self._test("INFO", re.search("INFO", out.getvalue()).group(0))

  # debug()
  def test_debug(self):
    out = StringIO()
    io.debug("foo", file=out)
    self._test("DEBUG", re.search("DEBUG", out.getvalue()).group(0))

  # warn()
  def test_warn(self):
    out = StringIO()
    io.warn("foo", file=out)
    self._test("WARN", re.search("WARN", out.getvalue()).group(0))

  # error()
  def test_error(self):
    out = StringIO()
    io.error("foo", file=out)
    self._test("ERROR", re.search("ERROR", out.getvalue()).group(0))

  # fatal()
  def test_fatal(self):
    out = StringIO()
    with self.assertRaises(SystemExit) as ctx:
      io.fatal("foo", file=out)
    self.assertEqual(ctx.exception.code, 1)
    self._test("ERROR", re.search("ERROR", out.getvalue()).group(0))
    self._test("fatal", re.search("fatal", out.getvalue()).group(0))

  def test_fatal_status(self):
    out = StringIO()
    with self.assertRaises(SystemExit) as ctx:
      io.fatal("foo", file=out, status=10)
    self.assertEqual(ctx.exception.code, 10)
    self._test("ERROR", re.search("ERROR", out.getvalue()).group(0))
    self._test("fatal", re.search("fatal", out.getvalue()).group(0))

  # prof()
  def test_prof(self):
    out = StringIO()
    io.prof("foo", file=out)
    self._test("PROF", re.search("PROF", out.getvalue()).group(0))
