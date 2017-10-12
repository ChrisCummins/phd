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

root := $(PWD)
cache := $(root)/.cache
UNAME := $(shell uname)
SHELL := /bin/bash
clean_targets =
distclean_targets =
test_targets =

# modules
include build/make/wget.make
include build/make/tar.make
include build/make/cmake.make
include build/make/ninja.make
include build/make/venv.make
include build/make/jupyter.make
include build/make/clgen.make
include build/make/cldrive.make
include build/make/clsmith.make
include build/make/clreduce.make

python_version = 3.6
python = $(venv_dir)/bin/python$(python_version)

all install: $(clsmith) $(clreduce) python protobuf

# run tests
.PHONY: test
test: $(test_targets)
	dsmith test


# python packages
python_packages = $(clgen) $(cldrive) $(jupyter)
python: $(venv_activate) $(python_packages) $(protobuf)
	./configure -r >/dev/null
	$(venv) pip install --only-binary=numpy '$(shell grep numpy requirements.txt)'
	$(venv) pip install -r requirements.txt
	$(venv) pip install '$(shell grep tensorflow requirements.txt)'
	$(venv) python ./setup.py install


# protocol buffers
protobuf = dsmith/dsmith_pb2.py
protobuf: $(python_packages) $(protobuf)

dsmith/dsmith_pb2.py: dsmith/protobuf/dsmith.proto python
	cd dsmith/protobuf && $(venv) protoc dsmith.proto --python_out=..


# launch Jupyter server
.PHONY: run
run: all
	$(venv) jupyter-notebook


# clean compiled files
.PHONY: clean
clean: $(clean_targets)
	$(venv) cd lib/clgen && make clean


# clean everything
.PHONY: distclean
distclean: $(distclean_targets)
	rm -rfv \
		$(venv_dir)/bin/cldrive \
		$(venv_dir)/bin/clgen \
		$(venv_dir)/bin/jupyter \
		.cache \
		.config.json \
		.config.make \
		clgen/_config.py \
		requirements.txt \
		$(NONE)


# help text
.PHONY: help
help:
	@echo "make {all,test,run,clean,distclean}"
