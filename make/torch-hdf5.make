# Makefile module for building torch-hdf5
#
# Usage:
#
#   * Add '$(torch-hdf5)' to list of prerequisites for files which require torch-hdf5.
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

# github repo
torch-hdf5_user := deepmind
torch-hdf5_version := 639bb4e62417ac392bf31a53cdd495d19337642b

torch-hdf5_url := https://github.com/$(torch-hdf5_user)/torch-hdf5/archive/$(torch-hdf5_version).zip
torch-hdf5_zip := $(cache)/$(torch-hdf5_user).torch-hdf5.$(torch-hdf5_version).zip
torch-hdf5_dir := $(root)/native/torch-hdf5/$(torch-hdf5_version)
torch-hdf5 := $(torch-hdf5_dir)/utils/prepare-builtins.o

# add this target as a prerequisite for files which require torch-hdf5
torch-hdf5: $(torch-hdf5)

$(torch-hdf5_zip):
	$(call wget,$@,$(torch-hdf5_url))

# unpack torch-hdf5 zip
$(torch-hdf5_dir)/LICENSE: $(torch-hdf5_zip)
	test -d $(dir $<)/torch-hdf5-$(torch-hdf5_version) || unzip -q $< -d $(dir $<)
	mkdir -p $(dir $(patsubst %/,%,$(dir $@)))
	mv $(dir $<)/torch-hdf5-$(torch-hdf5_version) $(patsubst %/,%,$(dir $@))
	touch $@

$(torch-hdf5): $(torch-hdf5_dir)/LICENSE $(torch)
	cd $(torch-hdf5_dir) && $(luarocks) make hdf5-0-0.rockspec

.PHONY: distclean-torch-hdf5
distclean-torch-hdf5:
	rm -rfv $(torch-hdf5_dir)
distclean_targets += distclean-torch-hdf5
