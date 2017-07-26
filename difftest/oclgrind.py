import subprocess
from labm8 import fs
from tempfile import NamedTemporaryFile

import clsmith
import clgen_mkharness
import clsmith_run_clreduce
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
    stdout, stderr = proc.communicate()

    print("STDERR:")
    print("\n".join([f">> {line}" for line in stderr.split("\n")]))

    if proc.returncode:
        return False

    if "Oclgrind: 1 errors generated" in stderr:
        return False

    return True


def oclgrind_verify_clgen(harness: CLgenHarness):
    with NamedTemporaryFile(prefix='oclgrind-harness-', delete=False) as tmpfile:
        binary_path = tmpfile.name
    try:
        clgen_mkharness.compile_harness(harness.src, binary_path, platform_id=0, device_id=0)

        # print("HARNESS:")
        # print(harness.src)
        return oclgrind_verify([binary_path])
    finally:
        fs.rm(binary_path)


def oclgrind_verify_clsmith(result: CLSmithResult):
    with NamedTemporaryFile(prefix='clsmith-kernel-', delete=False) as tmpfile:
        src_path = tmpfile.name
    try:
        with open(src_path, "w") as outfile:
            print(result.program.src, file=outfile)

        return oclgrind_verify(clsmith.cl_launcher_cli(src_path, 0, 0, timeout=None))
    finally:
        fs.rm(src_path)
