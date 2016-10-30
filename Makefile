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

# invoke make with CLGEN_GPU=0 to disable gpu
CLGEN_GPU ?= 1

# python configuration
PYTHON := python
VIRTUALENV := virtualenv
PIP := pip

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
include make/torch.make
include make/torch-rnn.make

native_targets := \
	clgen/data/bin/clang \
	clgen/data/bin/clang-format \
	clgen/data/bin/clgen-features \
	clgen/data/bin/clgen-rewriter \
	clgen/data/bin/llvm-config \
	clgen/data/bin/opt \
	clgen/data/bin/th \
	clgen/data/libclc \
	clgen/data/torch-rnn

native: $(native_targets)

clgen/data/bin/llvm-config: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/llvm-config $@
	touch $@

clgen/data/bin/clang: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/clang $@
	touch $@

clgen/data/bin/clang-format: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/clang-format $@
	touch $@

clgen/data/bin/opt: $(llvm)
	mkdir -p $(dir $@)
	ln -sf $(llvm_build)/bin/opt $@
	touch $@

clgen/data/bin/th: $(torch) $(torch_build)/bin/th
	mkdir -p $(dir $@)
	ln -sf $(torch_build)/bin/th $@
	touch $@

clgen/data/torch-rnn: $(torch-rnn)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(torch-rnn_dir) $@
	touch $@

clgen/data/libclc: $(libclc)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(libclc_dir)/generic/include $@
	touch $@

toolchain_flags := -xc++ $(llvm_CxxFlags) $(llvm_LdFlags)

clgen/data/bin/clgen-features: native/clgen-features.cpp $(llvm)
	mkdir -p $(dir $@)
	$(CXX) $< -o $@ $(toolchain_flags)

clgen/data/bin/clgen-rewriter: native/clgen-rewriter.cpp $(llvm)
	mkdir -p $(dir $@)
	$(CXX) $< -o $@ $(toolchain_flags)

# create virtualenv and install dependencies
virtualenv: env/bin/activate

# we keep GPU dependencies in a separate requirements file
ifeq ($(CLGEN_GPU),0)
env/bin/activate:
	$(VIRTUALENV) -p $(PYTHON) env
	$(env)pip install -r requirements.txt
	$(env)pip install -r requirements.devel.txt
else
env/bin/activate:
	$(VIRTUALENV) -p $(PYTHON) env
	$(env)pip install -r requirements.txt
	$(env)pip install -r requirements.devel.txt
	$(env)pip install -r requirements.gpu.txt
endif

# run tests
.PHONY: test
test: virtualenv $(native_targets)
	$(env)python ./setup.py install
	$(env)python ./setup.py test

# clean compiled files
.PHONY: clean
clean:
	rm -fv $(native_targets)

# clean everything
.PHONY: distclean distclean-virtualenv
distclean: $(distclean_targets)

distclean-virtualenv:
	rm -fr env

# install globally
.PHONY: install install-python install-native

ifeq ($(CLGEN_GPU),0)
install-python: install-native
	$(PIP) install -r requirements.txt
	$(PYTHON) ./setup.py install
else
install-python: install-native
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements.gpu.txt
	$(PYTHON) ./setup.py install
endif

install: install-python

.PHONY: docs-modules
docs-modules: install-python
	@echo "generating API documentation"
	cp docs/api.rst.template docs/api.rst
	@for module in $$(cd clgen; ls *.py | grep -v __init__.py); do \
		echo "adding module documentation for clgen.$${module%.py}"; \
		echo clgen.$${module%.py} >> docs/api.rst; \
		echo "$$(head -c $$(echo clgen.$${module%.py} | wc -c) < /dev/zero | tr '\0' '-')" >> docs/api.rst; \
		echo >> docs/api.rst; \
		echo ".. automodule:: clgen.$${module%.py}" >> docs/api.rst; \
		echo "   :members:" >> docs/api.rst; \
		echo "   :undoc-members:" >> docs/api.rst; \
		echo >> docs/api.rst; \
	done
	@echo "generating binary documentation"
	cp docs/binaries.rst.template docs/binaries.rst
	@for bin in $$(ls bin); do \
		echo "adding binary documentation for $$bin"; \
		echo $$bin >> docs/binaries.rst; \
		echo "$$(head -c $$(echo $$bin | wc -c) < /dev/zero | tr '\0' '-')" >> docs/binaries.rst; \
		echo >> docs/binaries.rst; \
		echo "::" >> docs/binaries.rst; \
		echo >> docs/binaries.rst; \
		./bin/$$bin --help | sed 's/^/    /' >> docs/binaries.rst; \
		echo >> docs/binaries.rst; \
	done

# generate documentation
.PHONY: docs
docs: docs-modules
	rm -rf docs/_build/html
	git clone git@github.com:ChrisCummins/clgen.git docs/_build/html
	cd docs/_build/html && git checkout gh-pages
	cd docs/_build/html && git reset --hard origin/gh-pages
	$(env3)$(MAKE) -C docs html

.PHONY: docs-publish
docs-publish: docs
	cd docs/_build/html && git add .
	cd docs/_build/html && git commit -m "Updated sphinx docs" || true
	cd docs/_build/html && git push -u origin gh-pages

# help text
.PHONY: help
help:
	@echo "make all        Compile code"
	@echo "make test       Run unit tests in virtualenv"
	@echo "make install    Install globally"
	@echo "make docs       Build documentation (performs partial install)"
	@echo "make clean      Remove compiled files"
	@echo "make distlcean  Remove all generated files"
	@echo
	@echo "set CLGEN_GPU=0 to disable CUDA support"
