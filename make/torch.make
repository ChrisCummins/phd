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
torch_remote := https://github.com/ChrisCummins/distro.git
#
# *** WARNING ***
#
# When changing torch_version, be sure to update ../install-deps.sh !
torch_version := 3467b980c56942451ee242937dbe76d15fcfc5ab
#

torch_src := $(root)/native/torch/$(torch_version)/src
torch_build := $(root)/native/torch/$(torch_version)/build
torch_deps := $(torch_src)/.bootstrapped
torch := $(torch_build)/.bootstrapped

# compiled binaries
luarocks := $(torch_build)/bin/luarocks
th := $(torch_build)/bin/th

torch: $(torch)

# clone git repo and submodules at a specific commit, then remove the .git dirs
$(torch_src)/install-deps:
	rm -rf $(torch_src)
	git clone $(torch_remote) $(torch_src)
	cd $(dir $@) && git reset --hard $(torch_version)
	cd $(dir $@) && git submodule update --init --recursive
	rm -rf $(torch_src)/.git
	touch $@

# torch dependencies
$(torch_deps): $(torch_src)/install-deps
	mkdir -p $(dir $@)
	@test -z "$$TRAVIS" || echo "Travis CI clang toolchain can't build openblas. Patching ..."
	@test -z "$$TRAVIS" || sed '/install_openblas /d' -i $<
	@test -z "$$TRAVIS" || echo "Travis CI is haning on install-deps. Skipping ..."
	@test -n "$$TRAVIS" || echo "cd $(torch_src) && bash install-deps"
	@test -n "$$TRAVIS" || (cd $(torch_src) && bash install-deps)
	touch $@

# if not configured with CUDA, then torch shouldn't use it
ifeq ($(USE_CUDA),0)
torch_disable_cuda := 1
else
torch_disable_cuda := 0
endif

# torch build
$(torch): $(torch_deps)
	mkdir -p $(dir $@)
	cd $(torch_src) && TORCH_NO_CUDA=$(torch_disable_cuda) TORCH_NO_RC=1 PREFIX="$(torch_build)" ./install.sh -b
	$(luarocks) install torch
	$(luarocks) install nn
	$(luarocks) install optim
	$(luarocks) install lua-cjson
	touch $@

.PHONY: distclean-torch
distclean-torch:
	rm -rf $(torch_build)
distclean_targets += distclean-torch
