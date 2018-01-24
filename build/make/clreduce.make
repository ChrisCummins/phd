# Makefile module for clreduce. Requires $(cmake) and $(ninja).
#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
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
# along with DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
clreduce_dir := $(root)/third_party/clreduce
clreduce := $(clreduce_dir)/build_creduce/creduce/creduce

clreduce: $(clreduce)

$(clreduce): cmake
	cd $(clreduce_dir) && PATH=$(dir $(ninja)):$$PATH make CMAKE=$(cmake)
all_targets += $(clreduce)

.PHONY: distclean-clreduce
distclean-clreduce:
	cd $(clreduce_dir) && make clean
distclean_targets += distclean-clreduce
