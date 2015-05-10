# Copyright (C) 2015 Chris Cummins.
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
import hashlib

def _sha1(data):
    return hashlib.sha1(data).hexdigest()

# Return the sha1 of string "data".
def sha1(data):
    return _sha1(data.encode("utf-8"))

# Return the sha1 of file at "path".
def sha1_file(path):
    return _sha1(open(path, 'rb').read())
