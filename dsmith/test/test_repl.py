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
import pytest

from dsmith import test
from dsmith import parser


def test_type_error():
    with pytest.raises(TypeError):
        parser.parse(2.5)


def test_unrecognized():
    with pytest.raises(ValueError):
        parser.parse("__ unrecognized input ___")

    with pytest.raises(parser.UnrecognizedInput):
        parser.parse("__ unrecognized input ___")


def test_empty():
    assert parser.parse("").msg == None
    assert parser.parse("").func == None


def test_hello():
    assert parser.parse(" Hi").msg == "Hi there!"
    assert parser.parse("hello").msg == "Hi there!"
    assert parser.parse("hello").func == None
