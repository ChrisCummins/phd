# Build DeepSmith
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
.DEFAULT_GOAL = all

# Fail if ./configure has not been run:
ifeq (,$(wildcard .config.make))
$(error Please run ./configure first)
endif
include .config.make

SHELL = /bin/bash

venv_dir := env/python3.6
venv_activate := $(venv_dir)/bin/activate
venv := source $(venv_activate) &&

python_version = 3.6
python = $(venv_dir)/bin/python$(python_version)

all: jupyter clgen cldrive CLSmith clreduce

clgen: $(venv_dir)/bin/clgen

# If CUDA is not available, build with NO_CUDA=1
ifeq ($(NO_CUDA),)
clgen_cuda_flag := --with-cuda
endif

$(venv_dir)/bin/clgen: $(venv_activate)
	$(venv) cd lib/clgen && ./configure -b $(clgen_cuda_flag)
	$(venv) cd lib/clgen && make
	$(venv) cd lib/clgen && make test

cldrive: $(venv_dir)/bin/cldrive

$(venv_dir)/bin/cldrive: $(venv_activate)
	$(venv) cd lib/cldrive && make install
	$(venv) cd lib/cldrive && make test

CLSmith: lib/CLSmith/build/bin/cl_launcher

lib/CLSmith/build/bin/cl_launcher:
	mkdir -pv lib/CLSmith/build
	cd lib/CLSmith/build && cmake .. -G Ninja
	cd lib/CLSmith/build && ninja
	cp -v lib/CLSmith/runtime/*.h lib/CLSmith/build/
	cp -v lib/CLSmith/build/*.h lib/CLSmith/runtime/

clreduce: lib/clreduce/build_creduce/creduce/creduce

lib/clreduce/build_creduce/creduce/creduce:
	cd lib/clreduce && make

jupyter: $(venv_dir)/bin/jupyter ~/.ipython/kernels/dsmith/kernel.json

$(venv_dir)/bin/jupyter: $(venv_activate)
	$(venv) pip install -r requirements.txt

~/.ipython/kernels/dsmith/kernel.json: build/ipython/kernels/dsmith/kernel.json
	mkdir -p ~/.ipython/kernels
	cp -Rv build/ipython/kernels/dsmith ~/.ipython/kernels

# make Jupyter kernel
build/ipython/kernels/dsmith/kernel.json: build/ipython/kernels/dsmith/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(PWD)/$(python)," -i $@

env/python3.6/bin/activate:
	virtualenv -p python3.6 env/python3.6

.PHONY: clean
clean:
	$(venv) cd lib/clgen && make clean
	rm -rfv \
		$(venv_dir)/bin/clgen \
		$(venv_dir)/bin/jupyter \
		$(venv_dir)/bin/cldrive \
		$(NONE)

.PHONY: run
run: all
	$(venv) jupyter-notebook

.PHONY: help
help:
	@echo "make {all,clean,run}"
	@echo
	@echo "If CUDA is not available, set NO_CUDA=1"
