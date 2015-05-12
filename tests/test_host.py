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
import labm8.host

import os
import socket

class TestHost(TestCase):

    def test_name(self):
        hostname = socket.gethostname()
        self._test(hostname, lab.host.HOSTNAME)
        self._test(hostname, lab.host.HOSTNAME)

    def test_pid(self):
        pid = os.getpid()
        self._test(pid, lab.host.PID)
        self._test(pid, lab.host.PID)

    def test_system(self):
        self._test(0, lab.host.system(["true"]))
        self._test(1, lab.host.system(["false"]))

if __name__ == '__main__':
    main()
