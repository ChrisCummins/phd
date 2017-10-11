# Makefile module for Jupyter. Requires $(venv) and $(python)
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
jupyter = $(venv_dir)/bin/jupyter

jupyter: $(jupyter) ~/.ipython/kernels/dsmith/kernel.json

$(jupyter): $(venv_activate)
	$(venv) pip install -r requirements.txt

~/.ipython/kernels/dsmith/kernel.json: build/ipython/kernels/dsmith/kernel.json
	mkdir -p ~/.ipython/kernels
	cp -Rv build/ipython/kernels/dsmith ~/.ipython/kernels

build/ipython/kernels/dsmith/kernel.json: build/ipython/kernels/dsmith/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(python)," -i $@
