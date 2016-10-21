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
# path to python3
PYTHON := python3
# path to virtualenv
VIRTUALENV := virtualenv


# Rules to create virtualenv:

# name of virtualenv (you can leave this)
VIRTUALENV_BASE := env

# create virtualenv
virtualenv: $(VIRTUALENV_BASE)/bin/activate
$(VIRTUALENV_BASE)/bin/activate:
	$(VIRTUALENV) -p $(PYTHON) $(VIRTUALENV_BASE)

# source virtualenv
env := source $(VIRTUALENV_BASE)/bin/activate &&


# Targets:

# run tests
test: virtualenv
	$(env)python ./setup.py test
