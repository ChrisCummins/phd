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
torch-rnnVersion := trunk
torch-rnnDir := $(PWD)/native/torch-rnn/$(torch-rnnVersion)
torch-rnn := $(torch-rnnBuild)/

# add this target as a prerequisite for files which require torch-rnn
torch-rnn: $(torch-rnn)

torch := $(PWD)/native/torch/trunk

$(torch):
	cd $(torch) && bash install-deps
	cd $(torch) && ./install.sh

$(torch-rnn): $(torch)
	luarocks install torch
	luarocks install nn
	luarocks install optim
	luarocks install lua-cjson
	cd $(PWD)/native/torch-hdf5/trunk && luarocks make hdf5-0.0.rockspec
	luarocks install cutorch
	luarocks install cunn

.PHONY: distclean-torch-rnn
distclean-torch-rnn:
	cd $(PWD)/native/torch/trunk && git clean -xfd
	cd $(PWD)/native/torch-hd5/trunk && git clean -xfd
	cd $(torch-rnnDir) && git clean -xfd
