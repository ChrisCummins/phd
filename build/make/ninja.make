# Makefile module for building ninja
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
ninja_version := 1.7.1
ninja_dir := $(root)/third_party/ninja/$(ninja_version)
ninja := $(ninja_dir)/ninja

ninja_tar := $(cache)/ninja.$(ninja_version).tar.gz
ninja_url := https://github.com/ninja-build/ninja/archive/v$(ninja_version).tar.gz

# add this target as a prerequisite for files which require ninja
ninja: $(ninja)

# fetch tarballs
$(ninja_tar):
	$(call wget,$@,$(ninja_url))

# unpack and build
$(ninja): $(ninja_tar)
	$(call unpack-tar,$(dir $@),$<)
	cd $(dir $@) && ./configure.py --bootstrap

.PHONY: distclean-ninja
distclean-ninja:
	rm -fr $(ninja_tar) $(ninja_dir)
distclean_targets += distclean-ninja
