# Build clgen
#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
.DEFAULT_GOAL = all

# configuration
ifeq (,$(wildcard .config.make))
$(error Please run ./configure first)
endif
include .config.make

root := $(PWD)
cache := $(root)/.cache
UNAME := $(shell uname)
clean_targets =
distclean_targets =

# allow overriding python:
PYTHON ?= python3
PIP ?= pip3

# We are using $(llvm-config --cxxflags) to include LLVM headers, which outputs
# clang-specific cflags, e.g. -fcolor-diagnostics.
CXX := clang++

# modules
include make/remote.make
include make/cmake.make
include make/gpuverify.make
include make/ninja.make
include make/llvm.make
include make/libclc.make
include make/oclgrind.make

data_symlinks = \
	$(root)/clgen/data/bin/clang \
	$(root)/clgen/data/bin/clang-format \
	$(root)/clgen/data/bin/llvm-config \
	$(root)/clgen/data/bin/opt \
	$(root)/clgen/data/libclc \
	$(root)/clgen/data/gpuverify

# FIXME: Oclgrind fails to build on Travis CI.
# See: https://travis-ci.org/ChrisCummins/clgen/builds/215785205
ifeq ($(TRAVIS),)
data_symlinks += $(root)/clgen/data/oclgrind
endif

data_bin = \
	$(root)/clgen/data/bin/clgen-features \
	$(root)/clgen/data/bin/clgen-rewriter

# build everything
all install: $(data_symlinks) $(data_bin)
	./configure -r >/dev/null
	$(PIP) install --only-binary=numpy '$(shell grep numpy requirements.txt)'
	$(PIP) install -r requirements.txt
	$(PYTHON) ./setup.py install

$(root)/clgen/data/bin/llvm-config: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/llvm-config $@
	touch $@

$(root)/clgen/data/bin/clang: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/clang $@
	touch $@

$(root)/clgen/data/bin/clang-format: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/clang-format $@
	touch $@

$(root)/clgen/data/bin/opt: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/opt $@
	touch $@

$(root)/clgen/data/libclc: $(libclc)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(libclc_dir)/generic/include $@
	touch $@

$(root)/clgen/data/gpuverify: $(gpuverify)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(root)/native/gpuverify/$(gpuverify_version) $@
	touch $@

$(root)/clgen/data/oclgrind: $(oclgrind)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(root)/native/oclgrind/$(oclgrind_version)/install $@
	touch $@

$(root)/clgen/data/bin/clgen-features: $(root)/native/clgen-features.cpp $(data_symlinks)
	mkdir -p $(dir $@)
	$(CXX) $< -o $@ $(llvm_CxxFlags) $(llvm_LdFlags)

$(root)/clgen/data/bin/clgen-rewriter: $(root)/native/clgen-rewriter.cpp $(data_symlinks)
	mkdir -p $(dir $@)
	$(CXX) $< -o $@ $(llvm_CxxFlags) $(llvm_LdFlags)

# run tests
.PHONY: test
test:
	clgen test

# clean compiled files
.PHONY: clean
clean: $(clean_targets)
	rm -fr $(data_symlinks) $(data_bin) corpus tests/data/tiny/corpus

# clean everything
.PHONY: distclean
distclean: $(distclean_targets)
	rm -f requirements.txt .config.json .config.make clgen/_config.py

# autogenerate documentation
.PHONY: docs-modules
docs-modules:
	@echo "generating API documentation"
	cp docs/api/.template docs/api/index.rst
	@for module in $$(cd clgen; env LC_COLLATE=C ls *.py | grep -v __init__.py); do \
		echo "adding module documentation for clgen.$${module%.py}"; \
		echo clgen.$${module%.py} >> docs/api/index.rst; \
		echo "$$(head -c $$(echo clgen.$${module%.py} | wc -c) < /dev/zero | tr '\0' '-')" >> docs/api/index.rst; \
		echo >> docs/api/index.rst; \
		echo ".. automodule:: clgen.$${module%.py}" >> docs/api/index.rst; \
		echo "   :members:" >> docs/api/index.rst; \
		echo "   :undoc-members:" >> docs/api/index.rst; \
		echo >> docs/api/index.rst; \
	done
	@echo "generating binary documentation"
	@cp docs/bin/.template docs/bin/index.rst
	@echo "adding binary documentation for clgen"
	@echo clgen >> docs/bin/index.rst
	@echo "$$(head -c $$(echo clgen | wc -c) < /dev/zero | tr '\0' '-')" >> docs/bin/index.rst
	@echo >> docs/bin/index.rst
	@echo "::" >> docs/bin/index.rst
	@echo >> docs/bin/index.rst
	@export CLGEN_AUTHOR='$$USER@$$HOSTNAME'
	@clgen --help | sed 's/^/    /' >> docs/bin/index.rst
	@echo >> docs/bin/index.rst

# generate documentation
.PHONY: docs
docs: docs-modules
	$(MAKE) -C docs html

# help text
.PHONY: help
help:
	@echo "make all        build CLgen"
	@echo "make test       run test suite (requires install)"
	@echo "make docs       build documentation (requires install)"
	@echo "make clean      remove compiled files"
	@echo "make distlcean  remove all generated files"
