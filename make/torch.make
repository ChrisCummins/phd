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

# git repo
torch_remote := https://github.com/torch/distro.git
torch_version := a58889e5289ca16b78ec7223dd8bbc2e01ef97e0

torch_src := $(root)/native/torch/$(torch_version)/src
torch_build := $(root)/native/torch/$(torch_version)/build
torch_deps := $(torch_src)/.bootstrapped
torch := $(torch_build)/.bootstrapped

# compiled binaries
luarocks := $(torch_build)/bin/luarocks
th := $(torch_build)/bin/th

torch: $(torch)

# clone git repo
$(torch_src)/install-deps:
	git clone $(torch_remote) $(torch_src)
	cd $(dir $@) && git checkout a58889e5289ca16b78ec7223dd8bbc2e01ef97e0
	cd $(dir $@) && git submodule update --init --recursive
	touch $@

# torch dependencies
$(torch_deps): $(torch_src)/install-deps
	mkdir -p $(dir $@)
	# Travis CI clang toolchain can't build openblas:
	test -z "$$TRAVIS" || sed '/install_openblas /d' -i $<
	# Travis CI seems to be hanging ?
	test -z "$$TRAVIS" || cd $(torch_src) && bash install-deps
	touch $@

# torch build
$(torch): $(torch_deps)
	mkdir -p $(dir $@)
	cd $(torch_src) && PREFIX="$(torch_build)" ./install.sh -b
	$(luarocks) install torch
	$(luarocks) install nn
	$(luarocks) install optim
	$(luarocks) install lua-cjson
	touch $@

.PHONY: distclean-torch
distclean-torch:
	rm -rf $(torch_build)
distclean_targets += distclean-torch
