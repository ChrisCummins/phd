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
import os
import pytest

from dsmith import test as tests
from dsmith import db


db.init(prefix="test_", engine_="sqlite", db_dir="/tmp/dsmith")


def test_add_client():
    client1 = db.Client(client="client1")
    client2 = db.Client(client="client2")

def test_add_client2():
    client1 = db.Client(client="client1")
