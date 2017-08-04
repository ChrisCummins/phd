import subprocess
from labm8 import fs
from tempfile import NamedTemporaryFile

import clsmith
import clgen_mkharness
from db import *

OCLGRIND = "../lib/clgen/native/oclgrind/c3760d07365b74ccda04cd361e1b567a6d99dd8c/install/bin/oclgrind"
assert fs.isexe(OCLGRIND)


def oclgrind_cli(timeout=60):
    """ runs the given path using oclgrind """
    return ['timeout', '-s9', str(timeout), OCLGRIND, '--max-errors', '1',
            '--uninitialized', '--data-races', '--uniform-writes',
            '--uniform-writes']


def oclgrind_verify(cmd):
    cmd = oclgrind_cli() + cmd

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    _, stderr = proc.communicate()

    if proc.returncode:
        return False

    if "Oclgrind: 1 errors generated" in stderr:
        return False

    return True


def verify_clsmith_testcase(testcase: CLSmithTestCase):
    with NamedTemporaryFile(prefix='clsmith-kernel-', delete=False) as tmpfile:
        src_path = tmpfile.name
    try:
        with open(src_path, "w") as outfile:
            print(testcase.program.src, file=outfile)

        return oclgrind_verify(clsmith.cl_launcher_cli(src_path, 0, 0, timeout=None))
    finally:
        fs.rm(src_path)


def verify_clgen_testcase(testcase: CLgenTestCase):
    with NamedTemporaryFile(prefix='oclgrind-harness-', delete=False) as tmpfile:
        binary_path = tmpfile.name
    try:
        clgen_mkharness.compile_harness(testcase.harness[0].src, binary_path, platform_id=0, device_id=0)

        return oclgrind_verify([binary_path])
    finally:
        fs.rm(binary_path)


def verify_testcase(session: session_t, tables: Tableset, testcase) -> bool:
    if testcase.oclverified == None:
        if tables.name == "CLSmith":
            testcase.oclverified = verify_clsmith_testcase(testcase)
        else:
            testcase.oclverified = verify_clgen_testcase(testcase)
        session.commit()

    return testcase.oclverified
