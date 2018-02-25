# Makefile module for CMake
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
cmake_version := 3.5.0
cmake_dir := $(root)/third_party/cmake/$(cmake_version)
cmake := $(cmake_dir)/bin/cmake

cmake_tar := $(cache)/cmake.$(cmake_version).tar.gz
ifeq ($(UNAME),Darwin)
cmake_url := http://cmake.org/files/v3.5/cmake-3.5.0-Darwin-x86_64.tar.gz
else
cmake_url := http://cmake.org/files/v3.5/cmake-3.5.0-Linux-x86_64.tar.gz
endif

# add this target as a prerequisite for files which require cmake
cmake: $(cmake)

# fetch tarballs
$(cmake_tar):
	$(call wget,$@,$(cmake_url),--no-check-certificate)

# unpack tar and update timestamp on target
ifeq ($(UNAME),Darwin)
$(cmake): $(cmake_tar)
	$(call unpack-tar,$(cmake_dir),$<,--strip-components=3)
	touch $@
else
$(cmake): $(cmake_tar)
	$(call unpack-tar,$(cmake_dir),$<)
	touch $@
endif

.PHONY: distclean-cmake
distclean-cmake:
	rm -fr $(cmake_tar) $(cmake_dir)
distclean_targets += distclean-cmake
