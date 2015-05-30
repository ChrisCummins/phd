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
import shutil


class Error(Exception):
    pass


def path(*components):
    """
    Get a file path.

    Concatenate all components into a path.
    """
    _path = os.path.join(*components)
    _path = os.path.expanduser(_path)

    return _path


def abspath(*components):
    """
    Get an absolute file path.

    Concatenate all components into an absolute path.
    """
    return os.path.abspath(path(*components))


def basename(path):
    """
    Return the basename of a given file path.
    """
    return os.path.basename(path)


def dirname(path):
    """
    Return the directory name of a given file path.
    """
    return os.path.dirname(path)


def is_subdir(child, parent):
    """
    Determine if "child" is a subdirectory of "parent". If child ==
    parent, returns True.
    """
    child_path = os.path.realpath(child)
    parent_path = os.path.realpath(parent)

    if len(child_path) < len(parent_path):
        return False

    for i in range(len(parent_path)):
        if parent_path[i] != child_path[i]:
            return False

    return True


# Directory history.
_cdhist = [os.getcwd()]


def cd(path):
    """
    Change working directory.

    Returns absolute path to new working directory.
    """
    path = abspath(path) # convert to absolute path
    _cdhist.append(path)
    os.chdir(path)
    return path


def cdpop():
    """
    Return the last directory.

    Returns absolute path to new working directory.
    """
    if len(_cdhist) > 1:
        _cdhist.pop() # remove current directory
        os.chdir(_cdhist[-1])
    return _cdhist[-1]


def pwd():
    """
    Return the path to the current working directory.
    """
    return _cdhist[-1]


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


def isdir(path):
    """
    Return whether a path exists, and is a directory.
    """
    return os.path.isdir(path)


def ls(p=".", abspaths=True):
    """
    List all files and directories in "path". If "abspaths", return
    absolute paths.
    """
    if abspaths:
        files = ls(p, abspaths=False)
        return [os.path.abspath(path(p, file)) for file in files]
    else:
        return os.listdir(p)


def rm(path):
    """
    Remove a file or directory.

    If path is a directory, this recursively removes the directory and
    any contents. Non-existent paths are silently ignored.

    Arguments:
        path (string): path to the file or directory to remove. May be
          absolute or relative.
    """
    if isfile(path):
        os.remove(path)
    elif exists(path):
        shutil.rmtree(path, ignore_errors=False)


def cp(src, dst):
    """
    Copy a file or directory.

    If source is a directory, this recursively copies the directory
    and its contents. If the destination is a directory, then this
    creates a copy of the source in the destination directory with the
    same basename.

    Arguments:

        src (string): path to the source file or directory.
        dst (string): path to the destination file or directory.

    Raises:

        IOError: if source does not exist.
    """
    if isdir(src):
        shutil.copytree(src, dst)
    elif isfile(src):
        shutil.copy(src, dst)
    else:
        raise IOError("Source '{0}' not found".format(src))


def mkdir(path, **kwargs):
    """
    Make directory "path", including any required parents. If
    directory already exists, do nothing.
    """
    if not isdir(path):
        os.makedirs(path, **kwargs)


def mkopen(p, *args, **kwargs):
    """
    A wrapper for the open() builtin which also ensures that the
    directory exists.
    """
    dir = os.path.dirname(p)
    mkdir(dir)
    return open(p, *args, **kwargs)


def read(path, rstrip=True, comment_char=None):
    """
    Read file "path" and return a list of lines. If comment_char is
    set, ignore the contents of lines following the comment_char.

    Raises:

        IOError: if reading path fails
    """
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
