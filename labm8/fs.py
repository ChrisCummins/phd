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

import os
import re

# Concatenate all components into a path.
def path(*components, abspath=True):
    relpath = "/".join(components)
    if abspath:
        return os.path.abspath(relpath)
    return relpath

# Change working directory.
def cd(path):
    os.chdir(path)

# Return the path to the current working directory.
def pwd():
    return os.getcwd()

# List all files and directories in "path". If "abspaths", return
# absolute paths.
def ls(p=".", abspaths=True):
    if abspaths:
        files = ls(p, abspaths=False)
        return [os.path.abspath(path(p, file)) for file in files]
    else:
        return os.listdir(p)

# Make directory "path", including any required parents. If directory
# already exists, do nothing.
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

# A wrapper for the open() builtin which also ensures that the
# directory exists.
def mkopen(p, *args, **kwargs):
    dir = os.path.dirname(p)
    mkdir(dir)
    return open(p, *args, **kwargs)

# Read file "path" and return a list of lines. If comment_char is set,
# ignore the contents of lines following the comment_char.
def read(path, rstrip=True, comment_char=None):
    has_comment_char = comment_char != None

    # Compile regexps.
    if has_comment_char:
        comment_line_re = re.compile("^\s*{char}".format(char=comment_char))
        not_comment_re = re.compile("[^{char}]+".format(char=comment_char))

    # Read file.
    file = open(path)
    lines = file.readlines()
    file.close()

    # Multiple definitions to handle all cases.
    if has_comment_char and rstrip:
        return [re.match(not_comment_re, line).group(0).rstrip()
                for line in lines
                if not re.match(comment_line_re, line)]
    elif has_comment_char:
        return [re.match(not_comment_re, line).group(0)
                for line in lines
                if not re.match(comment_line_re, line)]
    elif rstrip:
        return [line.rstrip() for line in lines]
    else:
        return lines
