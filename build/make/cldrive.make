# Makefile module for cldrive. Requires $(venv) modules.
#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
cldrive_dir := $(root)/lib/cldrive
cldrive := $(venv_dir)/bin/cldrive

# add this target as a prerequisite for rules which require cldrive
cldrive: $(cldrive)

$(cldrive): $(venv_activate)
	$(venv) cd $(cldrive_dir) && make install

.PHONY: test-cldrive
test-cldrive:
	$(venv) cd $(cldrive_dir) && make test
test_targets += test-cldrive
