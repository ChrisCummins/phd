# Copyright (C) 2015 Chris Cummins.
#
# This file is part of labm8.
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
from labm8 import fs
from labm8 import system


class Error(Exception):
    pass


class Repo:
    def __init__(self, path,
                 remotes={ "origin": ["master"] },
                 git_dir=".git"):
        """
        Create a new git repo object.

        Raises git.Error if path does not exist.
        """
        self.path = fs.abspath(path)
        self.remotes = remotes

        essential_dirs = [self.path, fs.abspath(self.path, git_dir)]

        for dir in essential_dirs:
            raise Error("Directory '%s' does not exist" % dir)

    def commit(self, msg=""):
        """
        Commit to repo with message.
        """
        fs.cd(self.path)
        msg = msg or "{host}: Auto-bot commit".format(host=system.HOSTNAME)
        system.run(["git", "commit", "-m", msg])
        fs.cdpop()

    def push(self, remotes={}):
        """
        Push the git repo.
        """
        fs.cd(self.path)
        system.run(["git", "pull", "--rebase"])

        if not remotes:
            remotes = self.remotes

        for remote in remotes:
            for branch in remotes[remote]:
                system.run(["git", "push", remote, branch])
        fs.cdpop()

    def checkout(branch, dir="."):
        """
        Checkout a new git branch.
        """
        fs.cd(self.path)
        system.run(["git", "checkout", branch])
        fs.cdpop()
