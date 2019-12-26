# A script which prints the python interpreter path and version.
#
# Use this to verify that bazel's python toolchain is set up correctly.
# Example:
#
#     $ bazel run //tools/py:python_version
#     /usr/bin/python3
#     3.7.5 (default, Nov 20 2019, 09:21:52)
#     [GCC 9.2.1 20191008]
#
import struct
import sys

print(sys.executable)
print(sys.version)
print(struct.calcsize("P") * 8, "bit")
