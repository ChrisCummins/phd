#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
from unittest import TestCase, main, skip
import tests

from labm8 import fs

import clgen
from clgen import explore

class TestExplore(TestCase):
    def test_explore(self):
        c = clgen.Corpus.from_json({
            "path": tests.data_path("tiny", "corpus", exists=False)
        })
        explore.explore(c.contentcache["kernels.db"])

    def test_explore_gh(self):
        db_path = tests.archive("tiny-gh.db")
        assert(fs.exists(db_path))

        explore.explore(db_path)


if __name__ == "__main__":
    main()
