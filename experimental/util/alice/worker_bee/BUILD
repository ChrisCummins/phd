# Copyright (c) 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# alice is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alice.  If not, see <https://www.gnu.org/licenses/>.

py_binary(
    name = "worker_bee",
    srcs = ["worker_bee.py"],
    deps = [
        "//experimental/util/alice:alice_pb_py_grpc",
        "//experimental/util/alice:bazel",
        "//experimental/util/alice:git_repo",
        "//labm8/py:app",
        "//labm8/py:system",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "worker_bee_test",
    srcs = ["worker_bee_test.py"],
    deps = [
        ":worker_bee",
        "//labm8/py:test",
    ],
)
