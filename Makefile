# Build DeepSmith
#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
all_targets =
clean_targets =
distclean_targets =
test_targets =

ifneq ($(root),$(ROOT))
$(error Directory moved from $(ROOT) to $(root). Please rerun ./configure)
endif

# modules
include tools/make/wget.make
include tools/make/tar.make
include tools/make/cmake.make
include tools/make/ninja.make
include tools/make/venv.make
include tools/make/jupyter.make
include tools/make/clgen.make

ifeq ($(WITH_OPENCL),1)
include tools/make/cldrive.make
include tools/make/clsmith.make
include tools/make/clreduce.make
include tools/make/llvm.make
endif

ifeq ($(WITH_GLSL),1)
include tools/make/glsl.make
endif

python_version = 3.6
python = $(venv_dir)/bin/python$(python_version)

all install: $(all_targets) python

# run tests
.PHONY: test
test: $(test_targets)
	dsmith test


# protocol buffers
protobuf = \
	dsmith/opencl/opencl_pb2.py \
	dsmith/dsmith_pb2.py \
	dsmith/dsmith_pb2_grpc.py

protobuf: $(python_packages) $(protobuf)

dsmith/opencl/opencl_pb2.py: dsmith/opencl/opencl.proto $(venv_activate)
	cd $(dir $@) && $(venv) protoc $(notdir $<) --python_out=.

dsmith/dsmith_pb2.py dsmith/dsmith_pb2_grpc.py: dsmith/protos/dsmith/dsmith.proto
	$(venv) python -m grpc_tools.protoc -Idsmith/protos --python_out=. --grpc_python_out=. dsmith/protos/dsmith/*.proto


# python packages
python_packages = $(clgen) $(cldrive) $(jupyter)
python: $(venv_activate) $(python_packages) $(protobuf)
	$(venv) ./configure -r >/dev/null
	$(venv) pip install --only-binary=numpy '$(shell grep numpy requirements.txt)'
	$(venv) pip install -r requirements.txt
	$(venv) pip install '$(shell grep tensorflow requirements.txt)'
	$(venv) python ./setup.py install


# launch Jupyter server
.PHONY: run
run: all
	$(venv) jupyter-notebook


# clean compiled files
.PHONY: clean
clean: $(clean_targets)
	rm -rfv tools/third_party/dsmith tools/dsmith/third_party/python*/site-packages/dsmith-*


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
