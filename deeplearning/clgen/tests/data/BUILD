# Test data for CLgen.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
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

filegroup(
    name = "cl_kernels",
    srcs = glob([
        "cl/*.cl",
        "cl/*.h",
        "cl/*.gs",
    ]),
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "dbs",
    srcs = glob(["db/*.db"]),
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "pico",
    srcs = [
        "pico/corpus.tar.bz2",
        "pico/model.json",
        "pico/sampler.json",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "tiny",
    srcs = [
        "tiny.tar.bz2",
        "tiny-gh.db.tar.bz2",
    ],
    visibility = ["//visibility:public"],
)
