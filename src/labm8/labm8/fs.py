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
import labm8 as lab

import os
import re
import os.path

class Watcher:
    """
    A Watcher is a filesystem monitor that is notified whenever one of
    it's member files is read or written to.
    """

    def __init__(self, path):
        """
        Create a new watcher object for the given path.
        """
        self._path = path

    def on_read(self, path):
        """
        File modified callback. Receives an absolute path to the read
        file.
        """
        pass

    def on_write(self, path):
        """
        File written callback. Receives an absolute path to the modified
        file.
        """
        pass

    def path(self):
        """
        Return the Watcher path.
        """
        return self._path

    def __str__(self):
        return "Watcher({0})".format(self.path())

def path(*args, **kwargs):
    """Get a file path.

    Concatenate all components into a path. Optional "abspath" keyword
    can be set to false to prevent real path expansion.
    """
    abspath = kwargs.pop('abspath', True)

    _path = os.path.join(*args)
    _path = os.path.expanduser(_path)
    if abspath: _path = os.path.abspath(_path)

    return _path

# Determine if "child" is a subdirectory of "parent". If child ==
# parent, returns True.
def is_subdir(child, parent):
    child_path = os.path.realpath(child)
    parent_path = os.path.realpath(parent)

    if len(child_path) < len(parent_path):
        return False

    for i in range(len(parent_path)):
        if parent_path[i] != child_path[i]:
            return False

    return True

# Change working directory.
def cd(path):
    os.chdir(path)

# Return the path to the current working directory.
def pwd():
    return os.getcwd()


def exists(path):
    """
    Return whether a file exists.
    """
    return os.path.exists(path)


def isfile(path):
    """
    Return whether a path exists, and is a file.
    """
    return os.path.isfile(path)


# List all files and directories in "path". If "abspaths", return
# absolute paths.
def ls(p=".", abspaths=True):
    if abspaths:
        files = ls(p, abspaths=False)
        return [os.path.abspath(path(p, file)) for file in files]
    else:
        return os.listdir(p)

# The global list of watchers.
_WATCHERS = set()

def register(watcher):
    """
    Register a new watcher object.
    """
    _WATCHERS.add(watcher)

def unregister(watcher):
    """
    Unregister a watcher object.
    """
    if watcher in _WATCHERS:
        _WATCHERS.remove(watcher)

def notified_watchers(path):
    """
    Return a list of watchers for the given path.
    """
    return set(filter(lambda x: is_subdir(path, x.path()), _WATCHERS))

def markread(path):
    """
    Mark a file as read.
    """
    abspath = os.path.abspath(path)
    listeners = notified_watchers(path)

    # Notify all listeners.
    for watcher in listeners:
        watcher.on_read(abspath)

    return path

def markwrite(path):
    """
    Mark a file as written (i.e. modified).
    """
    abspath = os.path.abspath(path)
    listeners = notified_watchers(path)

    # Notify all listeners.
    for watcher in listeners:
        watcher.on_write(abspath)

    return path

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
    file = open(markread(path))
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
