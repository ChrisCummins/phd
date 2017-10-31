# Makefile module for LLVM releases. Requires $(cmake) and $(ninja)
#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
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
llvm_dir := $(root)/lib/llvm

clang_362 := $(llvm_dir)/3.6.2/bin/clang
clang_371 := $(llvm_dir)/3.7.1/bin/clang
clang_381 := $(llvm_dir)/3.8.1/bin/clang
clang_391 := $(llvm_dir)/3.9.1/bin/clang
clang_401 := $(llvm_dir)/4.0.1/bin/clang
clang_500 := $(llvm_dir)/5.0.0/bin/clang
clang_trunk := $(llvm_dir)/trunk/bin/clang

clangs = $(NULL)
ifeq ($(WITH_LLVM_36),1)
clangs += $(clang_362)
endif
ifeq ($(WITH_LLVM_37),1)
clangs += $(clang_371)
endif
ifeq ($(WITH_LLVM_38),1)
clangs += $(clang_381)
endif
ifeq ($(WITH_LLVM_39),1)
clangs += $(clang_391)
endif
ifeq ($(WITH_LLVM_40),1)
clangs += $(clang_401)
endif
ifeq ($(WITH_LLVM_50),1)
clangs += $(clang_500)
endif
ifeq ($(WITH_LLVM_TRUNK),1)
clangs += $(clang_trunk)
endif

clangs: $(clangs)

all_targets += $(clangs)

$(clang_362):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/3.6.2
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/3.6.2/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/3.6.2/projects/compiler-rt
	cd $(llvm_dir)/.build/3.6.2 && git checkout release_36
	cd $(llvm_dir)/.build/3.6.2/tools/clang && git checkout release_36
	cd $(llvm_dir)/.build/3.6.2/projects/compiler-rt && git checkout release_36
	mkdir -pv $(llvm_dir)/.build/3.6.2/build
	cd $(llvm_dir)/.build/3.6.2/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/3.6.2/build && $(ninja)
	cd $(llvm_dir)/.build/3.6.2/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/3.6.2 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/3.6.2
	touch $@

$(clang_371):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/3.7.1
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/3.7.1/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/3.7.1/projects/compiler-rt
	cd $(llvm_dir)/.build/3.7.1 && git checkout release_37
	cd $(llvm_dir)/.build/3.7.1/tools/clang && git checkout release_37
	cd $(llvm_dir)/.build/3.7.1/projects/compiler-rt && git checkout release_37
	mkdir -pv $(llvm_dir)/.build/3.7.1/build
	cd $(llvm_dir)/.build/3.7.1/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/3.7.1/build && $(ninja)
	cd $(llvm_dir)/.build/3.7.1/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/3.7.1 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/3.7.1
	touch $@

$(clang_381):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/3.8.1
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/3.8.1/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/3.8.1/projects/compiler-rt
	cd $(llvm_dir)/.build/3.8.1 && git checkout release_36
	cd $(llvm_dir)/.build/3.8.1/tools/clang && git checkout release_36
	cd $(llvm_dir)/.build/3.8.1/projects/compiler-rt && git checkout release_36
	mkdir -pv $(llvm_dir)/.build/3.8.1/build
	cd $(llvm_dir)/.build/3.8.1/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/3.8.1/build && $(ninja)
	cd $(llvm_dir)/.build/3.8.1/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/3.8.1 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/3.8.1
	touch $@

$(clang_391):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/3.9.1
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/3.9.1/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/3.9.1/projects/compiler-rt
	cd $(llvm_dir)/.build/3.9.1 && git checkout release_39
	cd $(llvm_dir)/.build/3.9.1/tools/clang && git checkout release_39
	cd $(llvm_dir)/.build/3.9.1/projects/compiler-rt && git checkout release_39
	mkdir -pv $(llvm_dir)/.build/3.9.1/build
	cd $(llvm_dir)/.build/3.9.1/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/3.9.1/build && $(ninja)
	cd $(llvm_dir)/.build/3.9.1/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/3.9.1 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/3.9.1
	touch $@

$(clang_401):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/4.0.1
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/4.0.1/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/4.0.1/projects/compiler-rt
	cd $(llvm_dir)/.build/4.0.1 && git checkout release_40
	cd $(llvm_dir)/.build/4.0.1/tools/clang && git checkout release_40
	cd $(llvm_dir)/.build/4.0.1/projects/compiler-rt && git checkout release_40
	mkdir -pv $(llvm_dir)/.build/4.0.1/build
	cd $(llvm_dir)/.build/4.0.1/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/4.0.1/build && $(ninja)
	cd $(llvm_dir)/.build/4.0.1/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/4.0.1 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/4.0.1
	touch $@

$(clang_500):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/5.0.0
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/5.0.0/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/5.0.0/projects/compiler-rt
	cd $(llvm_dir)/.build/5.0.0 && git checkout release_50
	cd $(llvm_dir)/.build/5.0.0/tools/clang && git checkout release_50
	cd $(llvm_dir)/.build/5.0.0/projects/compiler-rt && git checkout release_50
	mkdir -pv $(llvm_dir)/.build/5.0.0/build
	cd $(llvm_dir)/.build/5.0.0/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/5.0.0/build && $(ninja)
	cd $(llvm_dir)/.build/5.0.0/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/5.0.0 -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/5.0.0
	touch $@

$(clang_trunk):
	mkdir -pv $(llvm_dir)/.build
	git clone http://llvm.org/git/llvm.git $(llvm_dir)/.build/trunk
	git clone http://llvm.org/git/clang.git $(llvm_dir)/.build/trunk/tools/clang
	git clone http://llvm.org/git/compiler-rt.git $(llvm_dir)/.build/trunk/projects/compiler-rt
	mkdir -pv $(llvm_dir)/.build/trunk/build
	cd $(llvm_dir)/.build/trunk/build && $(cmake) -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" ..
	cd $(llvm_dir)/.build/trunk/build && $(ninja)
	cd $(llvm_dir)/.build/trunk/build && $(cmake) -DCMAKE_INSTALL_PREFIX=$(llvm_dir)/trunk -P cmake_install.cmake
	rm -rf $(llvm_dir)/.build/trunk
	touch $@


.PHONE: distclean-llvm
distclean-llvm:
	rm -rfv $(llvm_dir)/3.* \
		$(llvm_dir)/4.* \
		$(llvm_dir)/5.* \
		$(llvm_dir)/trunk
distclean_targets += distclean-llvm
