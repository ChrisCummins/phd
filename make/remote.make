# Utilities for dealing with remote files
#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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

# download a file
#
# Arguments:
#   $1 (str): Target path
#   $2 (str): Source URL
#   $3 (str, optional): Additional arguments.
#
define wget
	mkdir -p $(dir $1)
	test -f $1 || wget -O $1 $2 $3
endef

# unpack a tarball
#
# Arguments:
#   $1 (str): Target directory
#   $2 (str): Source tarball.
#   $3 (str, optional): Additional arguments.
#
define unpack-tar
	mkdir -p $1
	tar -xf $2 -C $1 --strip-components=1 $3
endef
