# Copyright (C) 2015 Chris Cummins.
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

import labm8 as lab
from labm8 import system

import os
import socket

class TestSystem(TestCase):

    def test_name(self):
        hostname = socket.gethostname()
        self._test(hostname, system.HOSTNAME)
        self._test(hostname, system.HOSTNAME)

    def test_pid(self):
        pid = os.getpid()
        self._test(pid, system.PID)
        self._test(pid, system.PID)

    # Subprocess()
    def test_subprocess_stdout(self):
        p = system.Subprocess(["echo Hello"], shell=True)
        ret, out, err = p.run()
        self._test(0, ret)
        self._test("Hello\n", out)
        self._test("", err)

    def test_subprocess_stderr(self):
        p = system.Subprocess(["echo Hello >&2"], shell=True)
        ret, out, err = p.run()
        self._test(0, ret)
        self._test("", out)
        self._test("Hello\n", err)

    def test_subprocess_timeout(self):
        p = system.Subprocess(["sleep 10"], shell=True)
        self.assertRaises(system.SubprocessError, p.run, timeout=.1)

    def test_subprocess_timeout_pass(self):
        p = system.Subprocess(["true"], shell=True)
        ret, out, err = p.run(timeout=.1)
        self._test(0, ret)

    # run()
    def test_run(self):
        self._test((0, "", ""), system.run(["true"]))
        self._test((1, "", ""), system.run(["false"]))

    def test_run_timeout(self):
        self.assertRaises(system.SubprocessError, system.run,
                          ["sleep 10"], timeout=.1)
        self.assertRaises(system.SubprocessError, system.run,
                          ["sleep 10"], timeout=.1, num_attempts=2)

    def test_check_output(self):
        self._test("", system.check_output(["true"]))
        self.assertRaises(system.SubprocessError,
                          system.check_output, ["false"], exit_on_error=False)
        self._test("hello\n", system.check_output(["echo", "hello"]))


if __name__ == '__main__':
    main()
