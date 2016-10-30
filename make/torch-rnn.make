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
torch-rnn_version := trunk
torch-rnn_dir := $(PWD)/native/torch-rnn/$(torch-rnn_version)
torch-rnn := $(PWD)/native/torch-rnn/$(torch-rnn_version).bootstrapped

torch-rnn: $(torch-rnn)

# basic torch-rnn requirements
torch-rnn_base := $(PWD)/native/torch-rnn/$(torch-rnn_version).base.bootstrapped
$(torch-rnn_base): $(torch)
	$(luarocks) install torch
	$(luarocks) install nn
	$(luarocks) install optim
	$(luarocks) install lua-cjson
	cd $(PWD)/native/torch-hdf5/trunk && $(luarocks) make hdf5-0-0.rockspec
	touch $@

# Additional torch-rnn requirements for GPU support
torch_rnn_extra_rocks =
# TODO: it apperas that cltorch can no longer be installed using this method,
# but instead requires using a fork of the torch distro with OpenCL support
# baked in. It would be fairly substantial job to add support for this second
# torch distro, so I'm going to ignore it and simply disable OpenCL support
# for torch. See:
#   https://github.com/hughperkins/distro-cl
#
# ifeq ($(USE_OPENCL),1)
# torch_rnn_extra_rocks += cltorch clnn
# endif
ifeq ($(USE_CUDA),1)
torch_rnn_extra_rocks += cutorch cunn
endif

$(torch-rnn): $(torch-rnn_base)
	@for lib in $(torch_rnn_extra_rocks); do \
		echo "$(luarocks) install $$lib"; \
		$(luarocks) install $$lib; \
	done
	touch $@

.PHONY: distclean-torch-rnn
distclean-torch-rnn:
	cd $(PWD)/native/torch-hdf5/trunk && git clean -xfd
	cd $(torch-rnn_dir) && git clean -xfd
distclean_targets += distclean-torch-rnn
