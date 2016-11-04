# Workaround for Linux CUDA install.
#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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
.PHONY: cuda
ifeq ($(USE_CUDA),1)
cuda: cuda_headers cuda_libs
else
cuda:
endif

.PHONY: cuda_headers cuda_libs
cuda_headers: $(root)/make/patches/cuda-headers.sh
	$<

cuda_libs: $(root)/make/patches/cuda-libs.sh
	$<
