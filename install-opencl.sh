#!/usr/bin/env bash
#
# One-liner to install CLgen 0.0.24.
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
set -euv

version=0.0.24
major=$(echo $version | cut -d'.' -f1)
minor=$(echo $version | cut -d'.' -f2)
micro=$(echo $version | cut -d'.' -f3)

wget https://github.com/ChrisCummins/clgen/archive/$version.tar.gz -O clgen-$major.$minor.tar.gz
tar xf clgen-$version.tar.gz
rm clgen-$version.tar.gz
cd clgen-$major.$minor
./configure --batch --with-opencl
make
sudo -H make install
sudo -H make test
