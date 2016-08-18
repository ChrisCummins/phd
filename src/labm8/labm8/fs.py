# Copyright (C) 2015, 2016 Chris Cummins.
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
import os.path
import re
import shutil

from glob import iglob
from humanize import naturalsize

import labm8 as lab


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
_cdhist = []


def cd(path):
    """
    Change working directory.

    Returns absolute path to new working directory.
    """
    _cdhist.append(pwd())  # Push to history.
    path = abspath(path)
    os.chdir(path)
    return path


def cdpop():
    """
    Return the last directory.

    Returns absolute path to new working directory.
    """
    if len(_cdhist) >= 1:
        old = _cdhist.pop()  # Pop from history.
        os.chdir(old)
        return old
    else:
        return pwd()


def pwd():
    """
    Return the path to the current working directory.
    """
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


def isexe(path):
    """
    Return whether a path is an executable file.

    Arguments:

        path (str): Path of the file to check.

    Examples:

        >>> fs.isexe("/bin/ls")
        True

        >>> fs.isexe("/home")
        False

        >>> fs.isexe("/not/a/real/path")
        False

    Returns:

        bool: True if file is executable, else false.
    """
    return isfile(path) and os.access(path, os.X_OK)


def isdir(path):
    """
    Return whether a path exists, and is a directory.
    """
    return os.path.isdir(path)


def ls(root=".", abspaths=False, recursive=False):
    """
    Return a list of files in directory.

    Directory listings are sorted alphabetically. If the named
    directory is a file, return it's path.

    Examples:

        >>> fs.ls("foo")
        ["a", "b", "c"]

        >>> fs.ls("foo/a")
        ["foo/a"]

        >>> fs.ls("foo", abspaths=True)
        ["/home/test/foo/a", "/home/test/foo/b", "/home/test/foo/c"]

        >>> fs.ls("foo", recursive=True)
        ["a", "b", "b/d", "b/d/e", "c"]

    Arguments:

        root (str): Path to directory. Can be relative or absolute.
        abspaths (bool, optional): Return absolute paths if true.
        recursive (bool, optional): Recursively list subdirectories if
          true.

    Returns:

        list of str: A list of paths.

    Raises:

        OSError: If root directory does not exist.
    """
    def _expand_subdirs(file):
        if isdir(path(root, file)):
            return [file] + [path(file, x) for x in
                             ls(path(root, file), recursive=True)]
        else:
            return [file]

    if isfile(root):
        # If argument is a file, return path.
        return [abspath(root)] if abspaths else [basename(root)]
    elif abspaths:
        # Get relative names.
        relpaths = ls(root, recursive=recursive, abspaths=False)
        # Prepend the absolute path to each relative name.
        base = abspath(root)
        return [path(base, relpath) for relpath in relpaths]
    elif recursive:
        # Recursively expand subdirectories.
        paths = ls(root, abspaths=abspaths, recursive=False)
        return lab.flatten([_expand_subdirs(file) for file in paths])
    else:
        # List directory contents.
        return list(sorted(os.listdir(root)))


def lsdirs(root=".", **kwargs):
    """
    Return only subdirectories from a directory listing.

    Arguments:

        root (str): Path to directory. Can be relative or absolute.
        **kwargs: Any additional arguments to be passed to ls().

    Returns:

        list of str: A list of directory paths.

    Raises:

        OSError: If root directory does not exist.
    """
    paths = ls(root=root, **kwargs)
    if isfile(root):
        return []
    return [_path for _path in paths if isdir(path(root, _path))]


def lsfiles(root=".", **kwargs):
    """
    Return only files from a directory listing.

    Arguments:

        root (str): Path to directory. Can be relative or absolute.
        **kwargs: Any additional arguments to be passed to ls().

    Returns:

        list of str: A list of file paths.

    Raises:

        OSError: If root directory does not exist.
    """
    paths = ls(root=root, **kwargs)
    if isfile(root):
        return paths
    return [_path for _path in paths if isfile(path(root, _path))]


def rm(path, glob=True):
    """
    Remove a file or directory.

    If path is a directory, this recursively removes the directory and
    any contents. Non-existent paths are silently ignored.

    Supports Unix style globbing by default (disable using
    glob=False). For details on globbing pattern expansion, see:

        https://docs.python.org/2/library/glob.html

    Arguments:
        path (string): path to the file or directory to remove. May be
          absolute or relative. May contain unix glob
        glob (bool, optional): whether to perform Unix style pattern
          expansion of paths.
    """
    paths = iglob(path) if glob else [path]

    for file in paths:
        if isfile(file):
            os.remove(file)
        elif exists(file):
            shutil.rmtree(file, ignore_errors=False)


def cp(src, dst):
    """
    Copy a file or directory.

    If source is a directory, this recursively copies the directory
    and its contents. If the destination is a directory, then this
    creates a copy of the source in the destination directory with the
    same basename.

    If the destination already exists, this will attempt to overwrite
    it.

    Arguments:

        src (string): path to the source file or directory.
        dst (string): path to the destination file or directory.

    Raises:

        IOError: if source does not exist.
    """
    if isdir(src):
        # Overwrite an existing directory.
        if isdir(dst):
            rm(dst)
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
    return path


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
    ignore_comments = comment_char is not None

    file = open(path)
    lines = file.readlines()
    file.close()

    # Multiple definitions to handle all cases.
    if ignore_comments:
        comment_line_re = re.compile("^\s*{char}".format(char=comment_char))
        not_comment_re = re.compile("[^{char}]+".format(char=comment_char))

        if rstrip:
            # Ignore comments, and right strip results.
            return [re.match(not_comment_re, line).group(0).rstrip()
                    for line in lines
                    if not re.match(comment_line_re, line)]
        else:
            # Ignore comments, and don't strip results.
            return [re.match(not_comment_re, line).group(0)
                    for line in lines
                    if not re.match(comment_line_re, line)]
    elif rstrip:
        # No comments, and right strip results.
        return [line.rstrip() for line in lines]
    else:
        # Just a good old-fashioned read!
        return lines


def du(path, human_readable=True):
    """
    Get the size of a file in bytes or as a human-readable string.

    Arguments:

        path: Path to file.
        human_readable: If True, return a formatted string, e.g. "976.6 KiB"
    """
    if not exists(path):
        raise Error("file '{}' not found".format(path))
    size = os.stat(path).st_size
    if human_readable:
        return naturalsize(size)
    else:
        return size
