# Datastore implementations.
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "datastore",
    srcs = ["datastore.py"],
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_pb_py",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)

py3_image(
    name = "datastore_image",
    srcs = ["datastore.py"],
    base = "@base//image",
    main = "datastore.py",
    deps = [":datastore"],
)
