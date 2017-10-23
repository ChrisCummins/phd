#!/usr/bin/env bash
#
# One-liner to install CLgen 0.4.0.dev0.
#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
set -ex
wget https://github.com/ChrisCummins/clgen/archive/0.4.0.dev0.tar.gz -O clgen-0.4.0.dev0.tar.gz || {
    set +x
    echo >&2
    echo 'failed to fetch clgen archive.' >&2
    echo 'Please see install instructions:' >&2
    echo >&2
    echo '  <https://chriscummins.cc/clgen/>' >&2
    exit 1
}
tar xf clgen-0.4.0.dev0.tar.gz
rm clgen-0.4.0.dev0.tar.gz
cd clgen-0.4.0.dev0
./configure --batch --with-cuda
make

if [[ -n "$VIRTUAL_ENV" ]]; then
    # virtualen - no sudo required
    make all
    make test
    cd ..
    rm -rf clgen-0.4.0.dev0
else
    # system-wide - use sudo
    sudo -H make all
    sudo -H make test
    cd ..
    sudo rm -rf clgen-0.4.0.dev0
fi
echo "==> CLgen 0.4.0.dev0 installed"
