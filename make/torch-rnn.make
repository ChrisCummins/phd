# Makefile module for building torch-rnn
#
# Usage:
#
#   * Add '$(torch-rnn)' to list of prerequisites for files which require torch-rnn.
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
torch-rnn_user := ChrisCummins
torch-rnn_version := 078597f45b74e4bdce719e0afc4c68980b23ff67

torch-rnn_url := https://github.com/$(torch-rnn_user)/torch-rnn/archive/$(torch-rnn_version).zip
torch-rnn_zip := $(cache)/$(torch-rnn_user).torch-rnn.$(torch-rnn_version).zip
torch-rnn_dir := $(root)/native/torch-rnn/$(torch-rnn_version)
torch-rnn := $(root)/native/torch-rnn/$(torch-rnn_version).bootstrapped

# Additional torch-rnn requirements for GPU support
torch_rnn_rocks =
# TODO: it apperas that cltorch can no longer be installed using this method,
# but instead requires using a fork of the torch distro with OpenCL support
# baked in. It would be fairly substantial job to add support for this second
# torch distro, so I'm going to ignore it and simply disable OpenCL support
# for torch. See:
#   https://github.com/hughperkins/distro-cl
#
# ifeq ($(USE_OPENCL),1)
# torch_rnn_rocks += cltorch clnn
# endif
ifeq ($(USE_CUDA),1)
torch_rnn_rocks += cutorch cunn
endif

torch-rnn: $(torch-rnn)

# download torch-rnn zip
$(torch-rnn_zip):
	$(call wget,$@,$(torch-rnn_url))

# unpack torch-rnn zip
$(torch-rnn_dir)/train.lua: $(torch-rnn_zip)
	test -d $(dir $<)/torch-rnn-$(torch-rnn_version) || unzip -q $< -d $(dir $<)
	mkdir -p $(dir $(patsubst %/,%,$(dir $@)))
	mv $(dir $<)/torch-rnn-$(torch-rnn_version) $(patsubst %/,%,$(dir $@))
	touch $@

$(torch-rnn): $(torch-hdf5)
	@for lib in $(torch_rnn_rocks); do \
		echo "$(luarocks) install $$lib"; \
		$(luarocks) install $$lib; \
	done
	touch $@

.PHONY: distclean-torch-rnn
distclean-torch-rnn:
	rm -rf $(torch-rnn_dir)
distclean_targets += distclean-torch-rnn
