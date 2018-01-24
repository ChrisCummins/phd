# Makefile module for building libclc
#
# Usage:
#
#   * Add '$(libclc)' to list of prerequisites for files which require libclc.
#   * Add $(libclc_CxxFlags) to CFLAGS for files which require libclc.
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

# github repo
libclc_user := ChrisCummins
libclc_version := d0f8ca7247ded04afbf1561fc5823c3e3517d892

libclc_url := https://github.com/$(libclc_user)/libclc/archive/$(libclc_version).zip
libclc_zip := $(cache)/$(libclc_user).libclc.$(libclc_version).zip
libclc_dir := $(root)/native/libclc/$(libclc_version)
libclc := $(libclc_dir)/utils/prepare-builtins.o

# add this target as a prerequisite for files which require libclc
libclc: $(libclc)

# flags to build with compiled libclc
libclc_CxxFlags = \
	-Dcl_clang_storage_class_specifiers \
	-I$(libclcDir)/generic/include \
	-include $(libclcDir)/generic/include/clc/clc.h \
	-target nvptx64-nvidia-nvcl -x cl

$(libclc_zip):
	$(call wget,$@,$(libclc_url))

$(libclc_dir)/configure.py: $(libclc_zip)
	test -d $(dir $<)/libclc-$(libclc_version) || unzip -q $< -d $(dir $<)
	mkdir -p $(dir $(patsubst %/,%,$(dir $@)))
	rm -rf $(dir $@)
	mv $(dir $<)/libclc-$(libclc_version) $(patsubst %/,%,$(dir $@))
	touch $@

$(libclc): $(libclc_dir)/configure.py $(llvm)
	cd $(libclc_dir) && ./configure.py --with-llvm-config=$(llvm)
	cd $(libclc_dir) && $(MAKE)
	touch $@

.PHONY: distclean-libclc
distclean-libclc:
	rm -rf $(libclc_dir)
distclean_targets += distclean-libclc
