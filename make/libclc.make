# Makefile module for building libclc
#
# Usage:
#
#   * Add '$(libclc)' to list of prerequisites for files which require libclc.
#   * Add $(libclc_CxxFlags) to CFLAGS for files which require libclc.
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
libclcVersion := trunk
libclcDir := $(PWD)/native/libclc/$(libclcVersion)
libclc := $(libclcBuild)/utils/prepare-builtins.o

# add this target as a prerequisite for files which require libclc
libclc: $(libclc)

# flags to build with compiled libclc
libclc_CxxFlags = \
	-Dcl_clang_storage_class_specifiers \
	-I$(libclcDir)/generic/include \
	-include $(libclcDir)/generic/include/clc/clc.h \
	-target nvptx64-nvidia-nvcl -x cl

$(libclc): $(llvm)
	cd $(libclcDir) && ./configure.py --with-llvm-config=$(LlvmConfig)
	cd $(libclcDir) && $(MAKE)

.PHONY: distclean-libclc
distclean-libclc:
	cd $(libclcDir) && git clean -xfd
