# Makefile module for clgen. Requires $(venv)
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
clgen_dir := $(root)/lib/clgen
clgen := $(venv_dir)/bin/clgen
clgen: $(clgen)

# If CUDA is not available, build with NO_CUDA=1
ifeq ($(USE_CUDA),1)
clgen_cuda_flag := --with-cuda
endif

$(clgen): $(venv_activate)
	$(venv) cd $(clgen_dir) && ./configure -b $(clgen_cuda_flag)
	$(venv) cd $(clgen_dir) && make
	$(venv) cd $(clgen_dir) && make install


.PHONY: disttest-clgen
disttest-clgen: install
	$(venv) cd $(clgen_dir) && make test
disttest_targets += disttest-clgen
