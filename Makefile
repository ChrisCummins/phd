# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
.DEFAULT_GOAL = help

# allow overriding python:
PYTHON ?= python3
PIP ?= pip3

.PHONY: help install test docs

help:
	@echo "Usage: make {help,install,test,docs}"

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) ./setup.py install

test: install
	$(PYTHON) ./setup.py test

docs: install
	$(PIP) install -r docs/requirements.txt
	$(MAKE) -C docs html
