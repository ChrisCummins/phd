# Build clgen
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
.DEFAULT_GOAL = all

root := $(PWD)
distclean_targets =

# python configuration
PYTHON := python
VIRTUALENV := virtualenv
PIP := pip

# path to install native files
PREFIX := /usr/local

space :=
space +=

# source virtualenv
env := source env/bin/activate &&$(space)

# build everything
all: virtualenv native

# system name
UNAME := $(shell uname)

# modules
include make/remote.make
include make/cmake.make
include make/ninja.make
include make/llvm.make
include make/libclc.make
include make/torch-rnn.make

native := clgen/data/bin/clang native/clgen-rewriter $(libclc)

clgen/data/bin/clang: $(llvm)
	mkdir -p $(dir $@)
	ln -s $(PWD)/native/llvm/3.9.0/build/bin/clang $@

native: $(native)

rewriter_flags := $(CXXFLAGS) $(llvm_CxxFlags) $(LDFLAGS) $(llvm_LdFlags)

native/clgen-rewriter: native/clgen-rewriter.cpp $(llvm)
	$(CXX) $(rewriter_flags) $< -o $@

# create virtualenv and install dependencies
virtualenv: env/bin/activate

env/bin/activate:
	$(VIRTUALENV) -p $(PYTHON) env
	$(env)pip install -r requirements.txt
	$(env)pip install -r requirements.devel.txt

# run tests
.PHONY: test
test: virtualenv $(native)
	$(env)python ./setup.py install
	$(env)python ./setup.py test

# clean compiled files
.PHONY: clean
clean:
	rm -f native/clgen-rewriter

# clean everything
.PHONY: distclean distclean-virtualenv
distclean: $(distclean_targets)

distclean-virtualenv:
	rm -fr env

# install globally
.PHONY: install install-python install-native

install-python: install-native
	$(PIP) install -r requirements.txt
	$(PYTHON) ./setup.py install

install-native: $(native)
	cp native/clgen-rewriter $(PREFIX)/libexec

install: install-python install-native

# generate documentation
.PHONY: docs
docs: install-python
	rm -rf docs/modules
	mkdir -p docs/modules
	@for module in $$(cd clgen; ls *.py | grep -v __init__.py); do \
		cp -v docs/module.rst.template docs/modules/clgen.$${module%.py}.rst; \
		sed -i "s/@MODULE@/clgen.$${module%.py}/g" docs/modules/clgen.$${module%.py}.rst; \
		sed -i "s/@MODULE_UNDERLINE@/$$(head -c $$(echo clgen.$${module%.py} | wc -c) < /dev/zero | tr '\0' '=')/" docs/modules/clgen.$${module%.py}.rst; \
	done
	$(env3)$(MAKE) -C docs html

# help text
.PHONY: help
help:
	@echo "make all        Compile code"
	@echo "make test       Run unit tests in virtualenv"
	@echo "make install    Install globally"
	@echo "make docs       Build documentation (performs partial install)"
	@echo "make clean      Remove compiled files"
	@echo "make distlcean  Remove all generated files"
