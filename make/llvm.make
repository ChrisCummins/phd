# Makefile module for building LLVM
#
# Usage:
#
#   * Add '$(llvm)' to list of prerequisites for files which require LLVM.
#   * Add $(llvm_CxxFlags) to CFLAGS for files which are include LLVM headers.
#   * Add $(llvm_LdFlags) to link binaries against LLVM.
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
llvm_version := 3.9.0
llvm_src := $(root)/native/llvm/$(llvm_version)/src
llvm_build := $(root)/native/llvm/$(llvm_version)/build
llvm := $(llvm_build)/bin/llvm-config

# add this target as a prerequisite for files which require LLVM
llvm: $(llvm)

# flags to build with compiled LLVM
llvm_CxxFlags = \
	$(shell $(llvm) --cxxflags) \
	-isystem $(llvm_src)/include \
	-isystem $(llvm_build)/include \
	-isystem $(llvm_src)/tools/clang/include \
	-isystem $(llvm_build)/tools/clang/include \
	-std=c++11 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
	-D__STDC_LIMIT_MACROS -fno-rtti

# flags to link against compiled LLVM
llvm_LdFlags = \
	$(shell $(llvm) --system-libs) \
	-Wl,-rpath,$(llvm_build)/lib \
	-L$(llvm_build)/lib \
	-ldl \
	-lclangTooling \
	-lclangToolingCore \
	-lclangFrontend \
	-lclangDriver \
	-lclangSerialization \
	-lclangCodeGen \
	-lclangParse \
	-lclangSema \
	-lclangAnalysis \
	-lclangRewriteFrontend \
	-lclangRewrite \
	-lclangEdit \
	-lclangAST \
	-lclangLex \
	-lclangBasic \
	-lclang \
	-ldl \
	$(shell $(llvm) --libs) \
	-pthread \
	-lLLVMTarget -lLLVMMC \
	-lLLVMObject -lLLVMCore

ifeq ($(UNAME),Darwin)
llvm_LdFlags += -ldl -lcurses -lLLVMSupport -lcurses -ldl -lz
else
llvm_LdFlags += -ldl -lncurses -lLLVMSupport -lncurses -ldl -lz
endif

# LLVM components to download
llvm_components := llvm cfe clang-tools-extra

ifeq ($(UNAME),Darwin)
llvm_components += libcxx libcxxabi
endif

llvm_url_base := http://releases.llvm.org/$(llvm_version)/
llvm_url_suffix := -$(llvm_version).src.tar.xz
llvm_tars = $(addprefix $(cache)/,$(addsuffix $(llvm_url_suffix),$(llvm_components)))

# fetch LLVM tarballs
$(llvm_tars):
	$(call wget,$@,$(llvm_url_base)$(notdir $@))

# unpack an LLVM Tarball
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball
#
define unpack-llvm-tar
	$(call unpack-tar,$(llvm_src)/$1,$(cache)/$2$(llvm_url_suffix))
endef

# unpack LLVM tree from cached tarballs
ifeq ($(UNAME),Darwin)
$(llvm_src)/include/llvm/CMakeLists.txt: $(llvm_tars)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
	$(call unpack-llvm-tar,projects/libcxx,libcxx)
	$(call unpack-llvm-tar,projects/libcxxabi,libcxxabi)
	patch -p0 < make/patches/llvm/$(llvm_version)/FindAllSymbolsMain.cpp.patch
	mkdir -pv $(dir $@)
	touch $@
else
$(llvm_src)/include/llvm/CMakeLists.txt: $(llvm_tars)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
	patch -p0 < make/patches/llvm/$(llvm_version)/FindAllSymbolsMain.cpp.patch
	mkdir -pv $(dir $@)
	touch $@
endif

# flags to configure cmake build
llvm_cmake_flags := -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true \
	-DLLVM_TARGETS_TO_BUILD=X86 -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
	-DCLANG_ENABLE_ARCMT=OFF -DCMAKE_MAKE_PROGRAM=$(ninja) -G Ninja \
	-Wno-dev $(LLVM_CMAKE_FLAGS)

# build llvm
$(llvm): $(llvm_src)/include/llvm/CMakeLists.txt $(cmake) $(ninja)
	mkdir -p $(llvm_build)
	cd $(llvm_build) && $(cmake) $(llvm_src) $(llvm_cmake_flags)
	cd $(llvm_build) && $(ninja)
	mkdir -pv $(dir $@)
	touch $@

.PHONY: distclean-llvm
distclean-llvm:
	rm -rf $(llvm_build)
distclean_targets += distclean-llvm
