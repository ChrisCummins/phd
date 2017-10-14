#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
OpenCL test harnesses.
"""
from dsmith.langs import Harness
from dsmith.opencl.db import *


class Cl_launcher(Harness):
    __name__ = "cl_launcher"
    id = Harnesses.CLSMITH
    default_seed = 1
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        with ReuseSession(session) as s:
            return s.query(Threads)\
                .filter(Threads.gsize_x > 0).all()


class Cldrive(Harness):
    __name__ = "cldrive"
    id = Harnesses.DSMITH
    default_seed = 1
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        with ReuseSession(session) as s:
            return s.query(Threads)\
                .filter(Threads.gsize_x > 0).all()


class CompileOnly(Harness):
    __name__ = "clang"
