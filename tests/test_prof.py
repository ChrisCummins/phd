# Copyright (C) 2015, 2016 Chris Cummins.
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
from unittest import main
from tests import TestCase

import re

import labm8 as lab
from labm8 import prof

if lab.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


class TestProf(TestCase):

    def test_anonymous_timer(self):
        buf = StringIO()

        prof.start()
        prof.stop(file=buf)

        out = buf.getvalue()
        self._test("PROF", re.search("PROF", out).group(0))

    def test_named_timer(self):
        buf = StringIO()

        prof.start("foo")
        prof.stop("foo", file=buf)

        out = buf.getvalue()
        self._test("PROF", re.search("PROF", out).group(0))
        self._test(" foo: ", re.search(" foo: ", out).group(0))

    def test_named_timer(self):
        buf = StringIO()

        prof.start("foo")
        prof.start("bar")
        prof.stop("bar", file=buf)

        out = buf.getvalue()
        self._test("PROF", re.search("PROF", out).group(0))
        self._test(None, re.search(" foo: ", out))
        self._test(" bar: ", re.search(" bar: ", out).group(0))

        prof.stop("foo", file=buf)

        out = buf.getvalue()
        self._test("PROF", re.search("PROF", out).group(0))
        self._test(" foo: ", re.search(" foo: ", out).group(0))
        self._test(" bar: ", re.search(" bar: ", out).group(0))

    def test_new(self):
        t1 = prof.new()
        t2 = prof.new()
        self._test(True, t1 != t2)
        prof.stop(t1)
        prof.stop(t2)

    def test_reset(self):
        prof.start("foo")
        prof.reset("foo")
        prof.stop("foo")

    def test_stop_twice_error(self):
        prof.start("foo")
        prof.stop("foo")
        with self.assertRaises(prof.TimerNameError):
            prof.stop("foo")

    def test_stop_bad_name_error(self):
        with self.assertRaises(prof.TimerNameError):
            prof.stop("not a timer")

    def test_reset_bad_name_error(self):
        with self.assertRaises(prof.TimerNameError):
            prof.reset("not a timer")

    def test_unique_error(self):
        prof.start("foo")
        prof.start("foo")
        with self.assertRaises(prof.TimerNameError):
            prof.start("foo", unique=True)
