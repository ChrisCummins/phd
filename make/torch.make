# Makefile module for building torch
#
# Usage:
#
#   Add '$(torch)' to list of prerequisites for files which require torch.
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
torch_version := trunk
torch_src := $(PWD)/native/torch/$(torch_version)/src
torch_build := $(PWD)/native/torch/$(torch_version)/build
torch := $(torch_build)/.bootstrapped

torch: $(torch)

$(torch):
	mkdir -p $(dir $@)
	cd $(torch_src) && bash install-deps
	cd $(torch_src) && PREFIX="$(torch_build)" ./install.sh -b
	touch $@

.PHONY: distclean-torch
distclean-torch:
	rm -rf $(torch_build)
distclean_targets += distclean-torch
