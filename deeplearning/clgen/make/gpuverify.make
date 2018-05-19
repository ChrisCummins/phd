# Makefile module to pull GPUverify dependency
#
# Usage:
#
#   * Add '$(gpuverify)' to list of prerequisites for install targets.
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

# note that we've hardcoded a specific version of GPUverify, but the nightly
# build may update at any time and break this.
gpuverify_version := 2016-03-28
gpuverify_dir := $(root)/third_party/gpuverify
gpuverify := $(gpuverify_dir)/$(gpuverify_version)/gpuverify
gpuverify_url := http://multicore.doc.ic.ac.uk/tools/downloads/GPUVerifyLinux64-nightly.zip

gpuverify: $(gpuverify)

# we remove some python 2 scripts from bugle directory after unpacking, as
# setuptools discovers them and considers them part of CLgen, causing syntax
# errors
$(gpuverify):
	rm -rf $(gpuverify_dir)
	mkdir -p $(gpuverify_dir)
	cd $(gpuverify_dir) && wget $(gpuverify_url) -O gpuverify.zip
	cd $(gpuverify_dir) && unzip gpuverify.zip
	rm $(gpuverify_dir)/gpuverify.zip
	find $(gpuverify_dir)/$(gpuverify_version)/bugle/include-blang \
		-name '*.py' -exec rm -v {} \;
	touch $@

.PHONY: distclean-gpuverify
distclean-gpuverify:
	rm -rf $(gpuverify_dir)
distclean_targets += distclean-gpuverify
