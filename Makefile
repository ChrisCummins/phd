#
# Copyright (C) 2015-2017 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
# path to virtualenv
VIRTUALENV := virtualenv

space :=
space +=

# source virtualenvs
env3 := source env3/bin/activate &&$(space)
env2 := source env2/bin/activate &&$(space)

# create virtualenvs and install dependencies
virtualenv: env3/bin/activate env2/bin/activate

env3/bin/activate:
	$(VIRTUALENV) -p $(PYTHON3) env3
	$(env3)pip install -r requirements.txt
	$(env3)python ./setup.py install

env2/bin/activate:
	$(VIRTUALENV) -p $(PYTHON2) env2
	$(env2)pip install -r requirements.txt
	$(env2)python ./setup.py install

# run tests
.PHONY: test
test: virtualenv
	$(env3)python ./setup.py test
	$(env2)python ./setup.py test

# clean virtualenvs
.PHONY: clean
clean:
	rm -fr env3 env2

# install in virtualenv
.PHONY: install
install: virtualenv
	$(env3)pip install -r requirements.txt
	$(env3)python ./setup.py install
	$(env2)pip install -r requirements.txt
	$(env2)python ./setup.py install

# install globally
.PHONY: install-global
install-global:
	pip install -r requirements.txt
	python ./setup.py install

.PHONY: doc-modules
doc-modules:
	rm -rf docs/modules
	mkdir -p docs/modules
	@for module in $$(cd labm8; ls *.py | grep -v __init__.py); do \
		cp -v docs/module.rst.template docs/modules/labm8.$${module%.py}.rst; \
		sed -i "s/@MODULE@/labm8.$${module%.py}/g" docs/modules/labm8.$${module%.py}.rst; \
		sed -i "s/@MODULE_UNDERLINE@/$$(head -c $$(echo labm8.$${module%.py} | wc -c) < /dev/zero | tr '\0' '=')/" docs/modules/labm8.$${module%.py}.rst; \
	done

# generate documentation
.PHONY: docs
docs: doc-modules install-global
	$(env3)$(MAKE) -C docs html

# help text
.PHONY: help
help:
	@echo "make test      Run unit tests in virtualenv"
	@echo "make clean     Remove virtualenvs"
	@echo "make install   Install globally"
	@echo "make docs      Build documentation"
