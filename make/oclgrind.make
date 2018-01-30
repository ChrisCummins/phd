# Makefile module to pull OCLgrind dependency
#
# Usage:
#
#   * Add '$(oclgrind)' to list of prerequisites.
#   * Requires $(llvm), $(ninja), and $(cmake).
#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
oclgrind_version := c3760d07365b74ccda04cd361e1b567a6d99dd8c
oclgrind_remote := https://github.com/jrprice/Oclgrind.git
oclgrind_dir := $(root)/third_party/oclgrind/$(oclgrind_version)
oclgrind := $(oclgrind_dir)/install/bin/oclgrind

oclgrind: $(oclgrind)

oclgrind_cmake = $(cmake) .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_INSTALL_PREFIX=../install \
	-DLLVM_DIR=$(llvm_build)/lib/cmake/llvm \
	-DCLANG_ROOT=$(llvm_src)/tools/clang \
	-DCMAKE_CXX_FLAGS="-I$(llvm_build)/tools/clang/include" \
	-DCMAKE_MAKE_PROGRAM=$(ninja) -G Ninja

$(oclgrind):
	rm -rf $(root)/third_party/oclgrind
	mkdir -p $(root)/third_party/oclgrind
	cd $(root)/third_party/oclgrind && git clone $(oclgrind_remote) $(oclgrind_version)
	cd $(oclgrind_dir) && git reset --hard $(oclgrind_version)
	rm -rf $(oclgrind_dir)/.git
	mkdir $(oclgrind_dir)/build $(oclgrind_dir)/install
	cd $(oclgrind_dir)/build && $(oclgrind_cmake)
	cd $(oclgrind_dir)/build && $(ninja)
	cd $(oclgrind_dir)/build && $(ninja) test
	cd $(oclgrind_dir)/build && $(ninja) install

.PHONY: distclean-oclgrind
distclean-oclgrind:
	rm -rf $(root)/third_party/oclgrind
distclean_targets += distclean-oclgrind
