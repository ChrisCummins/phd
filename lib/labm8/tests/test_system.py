# Copyright (C) 2015-2017 Chris Cummins.
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
from unittest import main
from tests import TestCase

import getpass
import os
import socket

import labm8 as lab
from labm8 import fs
from labm8 import system


class TestSystem(TestCase):

    def test_hostname(self):
        hostname = socket.gethostname()
        self._test(hostname, system.HOSTNAME)
        self._test(hostname, system.HOSTNAME)

    def test_username(self):
        username = getpass.getuser()
        self._test(username, system.USERNAME)
        self._test(username, system.USERNAME)

    def test_uid(self):
        uid = os.getuid()
        self._test(uid, system.UID)
        self._test(uid, system.UID)

    def test_pid(self):
        pid = os.getpid()
        self._test(pid, system.PID)
        self._test(pid, system.PID)

    # ScpError
    def test_ScpError(self):
        err = system.ScpError("out", "err")
        self._test("out", err.out)
        self._test("err", err.err)
        self._test("out\nerr", err.__repr__())
        self._test("out\nerr", str(err))

    # Subprocess()
    def test_subprocess_stdout(self):
        p = system.Subprocess(["echo Hello"], shell=True)
        ret, out, err = p.run()
        self._test(0, ret)
        self._test("Hello\n", out)
        self._test(None, err)

    def test_subprocess_stderr(self):
        p = system.Subprocess(["echo Hello >&2"], shell=True)
        ret, out, err = p.run()
        self._test(0, ret)
        self._test(None, out)
        self._test("Hello\n", err)

    def test_subprocess_timeout(self):
        p = system.Subprocess(["sleep 10"], shell=True)
        self.assertRaises(system.SubprocessError, p.run, timeout=.1)

    def test_subprocess_timeout_pass(self):
        p = system.Subprocess(["true"], shell=True)
        ret, out, err = p.run(timeout=.1)
        self._test(0, ret)

    # run()
    def test_run(self):
        self._test((0, None, None), system.run(["true"]))
        self._test((1, None, None), system.run(["false"]))

    def test_run_timeout(self):
        self.assertRaises(system.SubprocessError, system.run,
                          ["sleep 10"], timeout=.1, shell=True)
        self.assertRaises(system.SubprocessError, system.run,
                          ["sleep 10"], timeout=.1, num_retries=2,
                          shell=True)

    # echo()
    def test_echo(self):
        system.echo("foo", "/tmp/labm8.tmp")
        self._test(["foo"], fs.read("/tmp/labm8.tmp"))
        system.echo("", "/tmp/labm8.tmp")
        self._test([""], fs.read("/tmp/labm8.tmp"))

    def test_echo_append(self):
        system.echo("foo", "/tmp/labm8.tmp")
        system.echo("bar", "/tmp/labm8.tmp", append=True)
        self._test(["foo", "bar"], fs.read("/tmp/labm8.tmp"))

    def test_echo_kwargs(self):
        system.echo("foo", "/tmp/labm8.tmp", end="_")
        self._test(["foo_"], fs.read("/tmp/labm8.tmp"))

    # sed()
    def test_sed(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        system.sed("Hello", "Goodbye", "/tmp/labm8.tmp")
        self._test(["Goodbye, world!"], fs.read("/tmp/labm8.tmp"))
        system.sed("o", "_", "/tmp/labm8.tmp")
        self._test(["G_odbye, world!"], fs.read("/tmp/labm8.tmp"))
        system.sed("o", "_", "/tmp/labm8.tmp", "g")
        self._test(["G__dbye, w_rld!"], fs.read("/tmp/labm8.tmp"))

    def test_sed_fail_no_file(self):
        self.assertRaises(system.SubprocessError, system.sed,
                          "Hello", "Goodbye", "/not/a/real/file")

    # which()
    def test_which(self):
        self._test("/bin/sh", system.which("sh"))
        self._test(None, system.which("not-a-real-command"))

    def test_which_path(self):
        self._test("/bin/sh", system.which("sh", path=("/usr", "/bin")))
        self._test(None, system.which("sh", path=("/dev",)))
        self._test(None, system.which("sh", path=("/not-a-real-path",)))
        self._test(None, system.which("not-a-real-command", path=("/bin",)))

    # scp()
    def test_scp(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Cleanup any existing file.
        fs.rm("/tmp/labm8.tmp.copy")
        self._test(False, fs.exists("/tmp/labm8.tmp.copy"))
        # Perform scp.
        system.scp("localhost", "/tmp/labm8.tmp", "/tmp/labm8.tmp.copy",
                   path="tests/bin")
        self._test(fs.read("/tmp/labm8.tmp"), fs.read("/tmp/labm8.tmp.copy"))

    def test_scp_user(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Cleanup any existing file.
        fs.rm("/tmp/labm8.tmp.copy")
        self._test(False, fs.exists("/tmp/labm8.tmp.copy"))
        # Perform scp.
        system.scp("localhost", "/tmp/labm8.tmp", "/tmp/labm8.tmp.copy",
                   path="tests/bin", user="test")
        self._test(fs.read("/tmp/labm8.tmp"), fs.read("/tmp/labm8.tmp.copy"))

    def test_scp_bad_path(self):
        # Error is raised if scp binary cannot be found.
        with self.assertRaises(system.CommandNotFoundError):
            system.scp("localhost", "/not/a/real/path", "/tmp/labm8.tmp.copy",
                       path="not/a/real/path")

    def test_scp_no_scp(self):
        # Error is raised if scp binary cannot be found.
        with self.assertRaises(system.CommandNotFoundError):
            system.scp("localhost", "/not/a/real/path", "/tmp/labm8.tmp.copy",
                       path="tests/data")

    def test_scp_bad_src(self):
        # Error is raised if source file cannot be found.
        with self.assertRaises(system.ScpError):
            system.scp("localhost", "/not/a/real/path", "/tmp/labm8.tmp.copy",
                       path="tests/bin")

    def test_scp_bad_dst(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Error is raised if destination file cannot be written.
        with self.assertRaises(system.ScpError):
            system.scp("localhost", "/tmp/labm8.tmp", "/not/a/valid/path",
                       path="tests/bin")

    def test_scp_bad_dst_permission(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Error is raised if no write permission for destination.
        with self.assertRaises(system.ScpError):
            system.scp("localhost", "/tmp/labm8.tmp", "/dev",
                       path="tests/bin")

    def test_scp_bad_host(self):
        # Error is raised if host cannot be found.
        with self.assertRaises(system.ScpError):
            system.scp("not-a-real-host", "/not/a/real/path",
                       "/tmp/labm8.tmp.copy",
                       path="tests/bin")

    # isprocess()
    def test_isprocess(self):
        self._test(True, system.isprocess(0))
        self._test(True, system.isprocess(os.getpid()))
        MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
        self._test(False, system.isprocess(MAX_PROCESSES + 1))
