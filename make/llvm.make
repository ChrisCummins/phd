# Makefile module for building LLVM
#
# Usage:
#
#   * Add '$(llvm)' to list of prerequisites for files which require LLVM.
#   * Add $(llvm_CxxFlags) to CFLAGS for files which are include LLVM headers.
#   * Add $(llvm_LdFlags) to link binaries against LLVM.
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
LlvmVersion := 3.9.0
LlvmSrc := $(PWD)/native/llvm/$(LlvmVersion)/src
LlvmBuild := $(PWD)/native/llvm/$(LlvmVersion)/build
LlvmLibDir := $(LlvmBuild)/lib
LlvmConfig := $(LlvmBuild)/bin/llvm-config
llvm := $(LlvmConfig)

# add this target as a prerequisite for files which require LLVM
llvm: $(llvm)

# flags to build with compiled LLVM
llvm_CxxFlags = \
	$(shell $(LlvmConfig) --cxxflags 2>/dev/null) \
	-isystem $(LlvmSrc)/include \
	-isystem $(LlvmBuild)/include \
	-isystem $(LlvmSrc)/tools/clang/include \
        -isystem $(LlvmBuild)/tools/clang/include \
        -std=c++11 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
        -D__STDC_LIMIT_MACROS -fno-rtti

# flags to link against compiled LLVM
llvm_LdFlags = \
	$(shell $(LlvmConfig) --system-libs 2>/dev/null) \
	-L$(LlvmBuild)/lib \
	-ldl \
	-lclangTooling \
	-lclangToolingCore \
	-lclangFrontend \
	-lclangDriver \
	-lclangSerialization \
	-lclangCodeGen \
	-lclangParse \
	-lclangSema \
	-lclangStaticAnalyzerFrontend \
	-lclangStaticAnalyzerCheckers \
	-lclangStaticAnalyzerCore \
	-lclangAnalysis \
	-lclangARCMigrate \
	-lclangRewriteFrontend \
	-lclangRewrite \
	-lclangEdit \
	-lclangAST \
	-lclangLex \
	-lclangBasic \
	-lclang \
	-ldl \
	$(shell $(LlvmConfig) --libs 2>/dev/null) \
	-pthread \
	-lLLVMTarget -lLLVMMC \
	-lLLVMObject -lLLVMCore

ifeq ($(UNAME),Darwin)
llvm_LdFlags += -ldl -lcurses -lLLVMSupport -lcurses -ldl
else
llvm_LdFlags += -ldl -lncurses -lLLVMSupport -lncurses -ldl
endif

# LLVM components to download
LlvmComponents := llvm cfe clang-tools-extra

ifeq ($(UNAME),Darwin)
LlvmComponents += compiler-rt libcxx libcxxabi
endif

LlvmUrlBase := http://llvm.org/releases/$(LlvmVersion)/
LlvmTar := -$(LlvmVersion).src.tar.xz
LlvmTarballs = $(addprefix native/llvm/$(LlvmVersion)/,$(addsuffix $(LlvmTar),$(LlvmComponents)))

# fetch LLVM tarballs
$(LlvmTarballs):
	$(call wget,$@,$(LlvmUrlBase)$(notdir $@))

# download a file
#
# Arguments:
#   $1 (str) Target path
#   $2 (str) Source URL
#
define wget
	mkdir -p $(dir $1)
	wget -O $1 $2
endef

# unpack a tarball
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball.
#   $3 (str) Tar arguments.
#
define unpack-tar
	mkdir -p $1
	tar -xf $2 -C $1 --strip-components=1
endef

# unpack an LLVM Tarball
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball
#
define unpack-llvm-tar
	$(call unpack-tar,$(LlvmSrc)/$1,native/llvm/$(LlvmVersion)/$2$(LlvmTar),-xf)
endef

# unpack LLVM tree from cached tarballs
ifeq ($(UNAME),Darwin)
$(LlvmSrc): $(LlvmTarballs)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
	$(call unpack-llvm-tar,projects/compiler-rt,compiler-rt)
	$(call unpack-llvm-tar,projects/libcxx,libcxx)
	$(call unpack-llvm-tar,projects/libcxxabi,libcxxabi)
else
$(LlvmSrc): $(LlvmTarballs)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
endif

# flags to configure cmake build
LlvmCMakeFlags := -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true \
	-DLLVM_TARGETS_TO_BUILD=X86 -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
	-DCLANG_ENABLE_ARCMT=OFF -G Ninja -Wno-dev

# Build rules.
ifeq ($(UNAME),Darwin)
$(llvm): $(LlvmSrc)
	rm -rf $(LlvmBuild)
	mkdir -p $(LlvmBuild)
	cd $(LlvmBuild) && cmake $(LlvmSrc) $(LlvmCMakeFlags) $(LLVM_CMAKE_FLAGS)
	cd $(LlvmBuild) && ninja
	cd $(LlvmBuild) && ninja cxx
else
$(llvm): $(LlvmSrc)
	rm -rf $(LlvmBuild)
	mkdir -p $(LlvmBuild)
	cd $(LlvmBuild) && cmake $(LlvmSrc) $(LlvmCMakeFlags) $(LLVM_CMAKE_FLAGS)
	cd $(LlvmBuild) && ninja
endif

.PHONY: distclean-llvm
distclean-llvm:
	rm -fv -r $(LlvmSrc)
	rm -fv -r $(LlvmBuild)
