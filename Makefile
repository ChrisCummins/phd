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
include build/make/llvm.make

python_version = 3.6
python = $(venv_dir)/bin/python$(python_version)

all install: $(clsmith) $(clreduce) python protobuf

# run tests
.PHONY: test
test: $(test_targets)
	dsmith test


# protocol buffers
protobuf = dsmith/opencl/opencl_pb2.py
protobuf: $(python_packages) $(protobuf)

dsmith/opencl/opencl_pb2.py: dsmith/opencl/opencl.proto $(venv_activate)
	cd $(dir $@) && $(venv) protoc $(notdir $<) --python_out=.


# data symlinks
data_symlinks = \
	$(root)/dsmith/data/bin/cl_launcher \
	$(root)/dsmith/data/bin/CLSmith \
	$(root)/dsmith/data/bin/clang-3.6.2 \
	$(root)/dsmith/data/bin/clang-3.7.1 \
	$(root)/dsmith/data/bin/clang-3.8.1 \
	$(root)/dsmith/data/bin/clang-3.9.1 \
	$(root)/dsmith/data/bin/clang-4.0.1 \
	$(root)/dsmith/data/bin/clang-5.0.0 \
	$(root)/dsmith/data/bin/clang-trunk \
	$(NULL)

$(root)/dsmith/data/bin/cl_launcher: $(cl_launcher)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/CLSmith: $(clsmith)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-3.6.2: $(clang_362)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-3.7.1: $(clang_371)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-3.8.1: $(clang_381)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-3.9.1: $(clang_391)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-4.0.1: $(clang_401)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-5.0.1: $(clang_501)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

$(root)/dsmith/data/bin/clang-trunk: $(clang_trunk)
	mkdir -p $(dir $@)
	ln -sf $< $@
	touch $@

.PHONY: clean-symlinks
clean-symlinks:
	rm -fv $(data_symlinks)
clean_targets += clean-symlinks


# header files
headers = \
	$(root)/dsmith/data/include/CLSmith.h

$(root)/dsmith/data/include/CLSmith.h: $(clsmith_include_dir)
	cp -v $</*.h $(dir $@)

.PHONY: clean-headers
clean-headers:
	rm -fv $(headers)
clean_targets += clean-headers


# python packages
python_packages = $(clgen) $(cldrive) $(jupyter)
python: $(venv_activate) $(python_packages) $(protobuf) $(data_symlinks) $(headers)
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
	rm -rfv build/lib/dsmith build/dsmith/lib/python*/site-packages/dsmith-*


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
