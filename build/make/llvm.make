# Makefile module for LLVM releases. Requires $(cmake) and $(ninja)
#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
llvm_dir := $(root)/lib/llvm

clang_362 := $(llvm_dir)/3.6.2/build/bin/clang
clang_371 := $(llvm_dir)/3.7.1/build/bin/clang
clang_381 := $(llvm_dir)/3.8.1/build/bin/clang
clang_391 := $(llvm_dir)/3.9.1/build/bin/clang
clang_401 := $(llvm_dir)/4.0.1/build/bin/clang
clang_500 := $(llvm_dir)/5.0.0/build/bin/clang
clang_trunk := $(llvm_dir)/trunk/build/bin/clang
clangs := \
	$(clang_362) \
	$(clang_371) \
	$(NULL)

clangs: $(clangs)

llvm_scripts = $(llvm_dir)/clone.sh $(llvm_dir)/checkout.sh $(llvm_dir)/build.sh

$(clang_362): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 3.6.2
	cd $(llvm_dir) && ./checkout.sh 3.6.2 release_36
	cd $(llvm_dir) && ./build.sh 3.6.2 3.6.2/build $(cmake) $(ninja)

$(clang_371): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 3.7.1
	cd $(llvm_dir) && ./checkout.sh 3.7.1 release_37
	cd $(llvm_dir) && ./build.sh 3.7.1 3.7.1/build $(cmake) $(ninja)

$(clang_381): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 3.8.1
	cd $(llvm_dir) && ./checkout.sh 3.8.1 release_38
	cd $(llvm_dir) && ./build.sh 3.8.1 3.8.1/build $(cmake) $(ninja)

$(clang_391): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 3.9.1
	cd $(llvm_dir) && ./checkout.sh 3.9.1 release_39
	cd $(llvm_dir) && ./build.sh 3.9.1 3.9.1/build $(cmake) $(ninja)

$(clang_401): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 4.0.1
	cd $(llvm_dir) && ./checkout.sh 4.0.1 release_40
	cd $(llvm_dir) && ./build.sh 4.0.1 4.0.1/build $(cmake) $(ninja)

$(clang_500): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh 5.0.0
	cd $(llvm_dir) && ./checkout.sh 5.0.0 release_50
	cd $(llvm_dir) && ./build.sh 5.0.0 5.0.0/build $(cmake) $(ninja)

$(clang_trunk): $(llvm_scripts)
	cd $(llvm_dir) && ./clone.sh trunk
	cd $(llvm_dir) && ./build.sh trunk trunk/build $(cmake) $(ninja)


.PHONE: distclean-llvm
distclean-llvm:
	rm -rfv $(root)/lib/llvm/3.6.2/build
distclean_targets += distclean-llvm
