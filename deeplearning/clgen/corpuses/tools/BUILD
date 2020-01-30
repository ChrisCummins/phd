# This package contains tools for working with CLgen corpuses.
#
# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.

py_binary(
    name = "export_preprocessed_files",
    srcs = ["export_preprocessed_files.py"],
    deps = [
        "//deeplearning/clgen/corpuses:preprocessed",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "export_preprocessed_files_test",
    srcs = ["export_preprocessed_files_test.py"],
    deps = [
        ":export_preprocessed_files",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
