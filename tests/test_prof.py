# Copyright (C) 2015-2017 Chris Cummins.
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
import os

import labm8 as lab
from labm8 import prof

if lab.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


class TestProf(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestProf, self).__init__(*args, **kwargs)
        os.environ["PROFILE"] = "1"

    def test_enable_disable(self):
        self.assertTrue(prof.is_enabled())
        prof.disable()
        self.assertFalse(prof.is_enabled())
        prof.enable()
        self.assertTrue(prof.is_enabled())

    def test_named_timer(self):
        buf = StringIO()

        prof.start("foo")
        prof.stop("foo", file=buf)

        out = buf.getvalue()
        self._test(" foo ", re.search(" foo ", out).group(0))

    def test_named_timer(self):
        buf = StringIO()

        prof.start("foo")
        prof.start("bar")
        prof.stop("bar", file=buf)

        out = buf.getvalue()
        self._test(None, re.search(" foo ", out))
        self._test(" bar ", re.search(" bar ", out).group(0))

        prof.stop("foo", file=buf)

        out = buf.getvalue()
        self._test(" foo ", re.search(" foo ", out).group(0))
        self._test(" bar ", re.search(" bar ", out).group(0))

    def test_stop_twice_error(self):
        prof.start("foo")
        prof.stop("foo")
        with self.assertRaises(KeyError):
            prof.stop("foo")

    def test_stop_bad_name_error(self):
        with self.assertRaises(KeyError):
            prof.stop("not a timer")

    def test_profile(self):
        def test_fn(x, y):
            return x + y

        self.assertEqual(prof.profile(test_fn, 1, 2), 3)

    def test_timers(self):
        x = len(list(prof.timers()))
        prof.start("new timer")
        self.assertEqual(len(list(prof.timers())), x + 1)
        prof.stop("new timer")
        self.assertEqual(len(list(prof.timers())), x)
