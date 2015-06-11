from __future__ import print_function

import re

import labm8 as lab
from labm8 import cache
from labm8 import crypto
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import system

import omnitune
from omnitune import util
from omnitune import llvm

if system.HOSTNAME != "tim":
    from omnitune import opencl
else:
    from omnitune import opencl_tim as opencl


class Error(Exception):
    """
    Module-level base error class.
    """
    pass


class FeatureExtractionError(Error):
    """
    Error thrown if feature extraction fails.
    """
    pass


def hash_kernel(north, south, east, west, max_wg_size, source):
    """
    Returns the hash of a kernel.
    """
    return crypto.sha1(".".join((str(north), str(south), str(east), str(west),
                                 str(max_wg_size), source)))


def hash_params(wg_c, wg_r):
    """
    Returns the hash of a set of parameter values.
    """
    return str(wg_c) + "x" + str(wg_r)


def hash_device(name, count):
    """
    Returns the hash of a device name + device count pair.
    """
    return str(count) + "x" + name.strip()


def hash_dataset(width, height, tin, tout):
    """
    Returns the hash of a data description.
    """
    return ".".join((str(width), str(height), tin, tout))


def hash_scenario(host, device_id, kernel_id, data_id):
    """
    Returns the hash of a scenario.
    """
    return crypto.sha1(".".join((host, device_id, kernel_id, data_id)))


def get_user_source(source):
    """
    Return the user source code for a stencil kernel.

    This strips the common stencil implementation, i.e. the border
    loading logic.

    Raises:
        FeatureExtractionError if the "end of user code" marker is not found.
    """
    lines = source.split("\n")
    user_source = []
    for line in lines:
        if line == "// --- SKELCL END USER CODE ---":
            return "\n".join(user_source)
        user_source.append(line)

    raise FeatureExtractionError("Failed to find end of user code marker")


def checksum_str(string):
    """
    Return the checksum for a string.
    """
    return crypto.sha1(string)


def get_source_features(checksum, source, path=""):
    user_source = skelcl.get_user_source(source)
    bitcode = llvm.bitcode(user_source, path=path)
    instcounts = llvm.instcounts(bitcode, path=path)
    ratios = llvm.instcounts2ratios(instcounts)

    return vectorise_ratios(checksum, source, ratios)


def get_kernel_name_and_type(source):
    """
    Figure out whether a kernel is synthetic or otherwise.

    Arguments:

        source (str): User source code for kernel.

    Returns:

        (bool, str): Where bool is whether the kernel is synthetic,
          and str is the name of the kernel. If it can't figure out
          the name, returns "unknown".
    """
    def _get_printable_source(lines):
        for i,line in enumerate(lines):
            # Store index of shape define lines.
            if re.search(r"^#define SCL_NORTH", line): north = i
            if re.search(r"^#define SCL_SOUTH", line): south = i
            if re.search(r"^#define SCL_EAST",  line): east  = i
            if re.search(r"^#define SCL_WEST",  line): west  = i

            # If we've got as far as the user function, then print
            # what we have.
            if re.search("^(\w+) USR_FUNC", line):
                return "\n".join([
                    lines[north],
                    lines[south],
                    lines[east],
                    lines[west],
                    ""
                ] + lines[i:])

        # Fall through, just print the whole bloody lot.
        return "\n".join(lines)

    lines = source.split("\n")

    # Look for clues in the source.
    for line in lines:
        if re.search('^// "Simple" kernel', line):
            return True, "simple"
        if re.search('^// "Complex" kernel', line):
            return True, "complex"

    # Base case, prompt the user.
    print("\nFailed to automatically deduce a kernel name and type.")
    print("Resorting to help from meat space:")
    print("***************** BEGIN SOURCE *****************")
    print(_get_printable_source(lines), "\n")
    name = raw_input("Name me: ")
    synthetic = raw_input("Synthetic? (y/n): ")

    # Sanitise and return user input
    return (True if synthetic.strip().lower() == "y" else False,
            name.strip().lower())


def main():
    import proxy
    proxy.main()
