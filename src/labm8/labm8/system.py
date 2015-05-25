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

"""
System module.

Variables:
  HOSTNAME System hostname.
  PID Process ID.
"""

import os
import socket
import subprocess
import threading

import labm8 as lab
from labm8 import fs
from labm8 import io

HOSTNAME = socket.gethostname()

PID = os.getpid()


class Error(Exception):
    pass


class SubprocessError(Error):
    """
    Error thrown if a subprocess fails.
    """
    pass


class Subprocess(object):
    """
    Subprocess abstraction.

    Wrapper around subprocess.Popen() which provides the ability to
    force a timeout after a number of seconds have elapsed.
    """
    def __init__(self, cmd, shell=False,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,
                 decode_out=True):
        """
        Create a new subprocess.
        """
        self.cmd = cmd
        self.process = None
        self.stdout = None
        self.stderr = None
        self.shell = shell
        self.decode_out = decode_out

        self.stdout_dest = stdout
        self.stderr_dest = stderr

    def run(self, timeout=-1):
        """
        Run the subprocess.

        Arguments:
            timeout (optional) If a positive real value, then timout after
                the given number of seconds.

        Raises:
            SubprocessError If subprocess has not completed after "timeout"
                seconds.
        """
        def target():
            self.process = subprocess.Popen(self.cmd,
                                            stdout=self.stdout_dest,
                                            stderr=self.stderr_dest,
                                            shell=self.shell)
            self.stdout, self.stderr = self.process.communicate()

            # Decode output if required by the user.
            if self.decode_out:
                self.stdout = self.stdout.decode("utf-8")
                self.stderr = self.stderr.decode("utf-8")

        thread = threading.Thread(target=target)
        thread.start()

        if timeout > 0:
            thread.join(timeout)
            if thread.is_alive():
                self.process.terminate()
                thread.join()
                raise SubprocessError(("Reached timeout after {t} seconds"
                                       .format(t=timeout)))
        else:
            thread.join()

        return self.process.returncode, self.stdout, self.stderr



def run(args, num_attempts=1, timeout=-1, **kwargs):
    """
    Run "args", redirecting stdout and stderr to "out". Returns exit
    status.
    """
    for i in range(num_attempts):
        try:
            process = Subprocess(args, **kwargs)
            return process.run(timeout)
        except SubprocessError:
            pass
        except AttributeError:
            pass

    raise SubprocessError("Failed after {i} attempts".format(i=i))


def check_output(args, shell=False, exit_on_error=True):
    """Run "args", returning stdout and stderr.

    If the process fails, raises a SubprocessError. If "exit_on_error"
    is True, execution halts.
    """
    try:
        output = subprocess.check_output(args, shell=shell,
                                         stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as err:
        msg = ("Subprocess '%s' failed with exit code '{}'"
               % err.cmd, err.returncode)

        if exit_on_error:
            io.error(err.output)
            io.fatal(msg, err.returncode)
        raise SubprocessError(msg)


def sed(match, replacement, path, modifiers=""):
    """
    Perform sed text substitution.
    """
    cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

    os.system(cmd)
