# Makefile module for CLSmith. Requires $(cmake) and $(ninja) modules.
#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
clsmith_dir := $(root)/third_party/clsmith
clsmith := $(clsmith_dir)/build/CLSmith
cl_launcher := $(clsmith_dir)/build/cl_launcher
clsmith_include_dir := $(clsmith_dir)/runtime

# add this target as a prerequisite for rules which require CLSmith
clsmith: $(clsmith)

$(clsmith): cmake ninja
	mkdir -pv $(clsmith_dir)/build
	cd $(clsmith_dir)/build && $(cmake) .. -G Ninja -DCMAKE_MAKE_PROGRAM=$(ninja)
	cd $(clsmith_dir)/build && $(ninja)
	cp -v $(clsmith_dir)/runtime/*.h $(clsmith_dir)/build/
	cp -v $(clsmith_dir)/build/*.h $(clsmith_dir)/runtime/
all_targets += $(clsmith)

.PHONY: distclean-clsmith
distclean-clsmith:
	rm -rfv $(clsmith_dir)/build
distclean_targets += distclean-clsmith
