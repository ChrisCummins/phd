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
PYTHON ?= python
PIP ?= pip

# modules
include make/remote.make
include make/cuda.make
include make/torch.make
include make/torch-hdf5.make
include make/cmake.make
include make/ninja.make
include make/llvm.make
include make/libclc.make
include make/torch-rnn.make

data_symlinks = \
	$(root)/clgen/data/bin/clang \
	$(root)/clgen/data/bin/clang-format \
	$(root)/clgen/data/bin/llvm-config \
	$(root)/clgen/data/bin/opt \
	$(root)/clgen/data/libclc \
	$(root)/clgen/data/torch-rnn

data_bin = \
	$(root)/clgen/data/bin/clgen-features \
	$(root)/clgen/data/bin/clgen-rewriter


# build everything
all: $(data_symlinks) $(data_bin)

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

$(root)/clgen/data/torch-rnn: $(torch-rnn)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(torch-rnn_dir) $@
	touch $@

$(root)/clgen/data/libclc: $(libclc)
	mkdir -p $(dir $@)
	rm -f $@
	ln -sf $(libclc_dir)/generic/include $@
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
	$(PYTHON) ./setup.py test

# clean compiled files
.PHONY: clean
clean: $(clean_targets)
	rm -fr $(data_symlinks) $(data_bin) corpus tests/data/tiny/corpus

# clean everything
.PHONY: distclean
distclean: $(distclean_targets)
	rm -f requirements.txt .config.json .config.make clgen/config.py

# install CLgen
.PHONY: install
install: cuda
	$(PIP) install --upgrade pip
	$(PIP) install --only-binary=numpy 'numpy>=1.10.4'
	$(PIP) install --only-binary=scipy 'scipy>=0.16.1'
	$(PIP) install --only-binary=pandas 'pandas>=0.19.0'
	$(PIP) install 'Cython==0.23.4'
	$(PIP) install -r requirements.txt
	$(PYTHON) ./setup.py install

# autogenerate documentation
.PHONY: docs-modules
docs-modules:
	@echo "generating API documentation"
	cp docs/api/.template docs/api/index.rst
	@for module in $$(cd clgen; ls *.py | grep -v __init__.py); do \
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
	cp docs/bin/.template docs/bin/index.rst
	@for bin in $$(ls bin); do \
		echo "adding binary documentation for $$bin"; \
		echo $$bin >> docs/bin/index.rst; \
		echo "$$(head -c $$(echo $$bin | wc -c) < /dev/zero | tr '\0' '-')" >> docs/bin/index.rst; \
		echo >> docs/bin/index.rst; \
		echo "::" >> docs/bin/index.rst; \
		echo >> docs/bin/index.rst; \
		export CLGEN_AUTHOR='$$USER@$$HOSTNAME'; \
		./bin/$$bin --help | sed 's/^/    /' >> docs/bin/index.rst; \
		echo >> docs/bin/index.rst; \
	done

# generate documentation
.PHONY: docs
docs: docs-modules
	rm -rf docs/_build/html
	git clone git@github.com:ChrisCummins/clgen.git docs/_build/html
	cd docs/_build/html && git checkout gh-pages
	cd docs/_build/html && git reset --hard origin/gh-pages
	$(env3)$(MAKE) -C docs html

# publish documentation
.PHONY: docs-publish
docs-publish: docs
	cd docs/_build/html && git add .
	cd docs/_build/html && git commit -m "Updated sphinx docs" || true
	cd docs/_build/html && git push -u origin gh-pages

# help text
.PHONY: help
help:
	@echo "make all        build CLgen"
	@echo "make install    install CLgen"
	@echo "make test       run test suite (requires install)"
	@echo "make docs       build documentation (requires install)"
	@echo "make clean      remove compiled files"
	@echo "make distlcean  remove all generated files"
