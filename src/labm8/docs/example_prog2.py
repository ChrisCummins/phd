#!/usr/bin/env python
#
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
"""
In this example script, the user wants to collect performance
information from multiple benchmarks compiled under differing
conditions.

**** NOTE that this is not (yet) valid labm8, but is used to help in
the design process!! ****
"""

import re

import labm8 as lab
from labm8 import git
from labm8 import make
from labm8 import result
from labm8 import static
from labm8 import system
from labm8 import testcase
from labm8 import variables


class E14(tescase.Harness):
    """
    Experiment harness.
    """
    def setup(self):
        pass

    def teardown(self):
        pass


class SimpleBigTest(testcase.Testcase):
    """
    Experiment test case.
    """

    ROOT = fs.path("~/src/msc-thesis/skelcl")
    BUILD_ROOT = fs.path(ROOT, "build/examples/SimpleBig")
    SRC = fs.path(ROOT, "examples/SimpleBig/main.cpp")

    def setup(self):
        """
        Testcase setup.

        Set the correct border size, then build the benchmark.
        """
        border = self.invars["border"]

        # Set the border size.
        system.sed("(define NORTH) [0-9]+", "\\1 {}" % border[0],
                   self.SRC)
        system.sed("(define WEST) [0-9]+", "\\1 {}" % border[1],
                   self.SRC)
        system.sed("(define SOUTH) [0-9]+", "\\1 {}" % border[2],
                   self.SRC)
        system.sed("(define EAST) [0-9]+", "\\1 {}" % border[3],
                   self.SRC)

        # Build benchmark.
        fs.cd(BUILD_ROOT)
        make.clean()
        make.target("SimpleBig")

    def sample(self):
        """
        Collect a sample.

        Returns the program output as a string and the returncode as
        an integer.
        """
        output, returncode = system.run("./SimpleBig")
        return {"output": output, "returncode": returncode}

def invars_are_legal(invars):
    """
    Return if invars are legal.

    Check whether the given invars are legal for execution on this
    device.
    """
    # Check that we're on the right host.
    host = invars["host"]
    if host != host.HOSTNAME:
        return False

    # Check that we have enough devices to execute.
    device_type = invars["device"][0]
    device_count = invars["device"][1]
    matching_devices = [x for x in host.OPENCL_DEVICES if x.type == device_type]
    if len(matching_devices) < device_count:
         return False

    # Check that the workgroup size is not larger than imposed by
    # hardware constraints.
    wg_size_c = invars["workgroup size c"]
    wg_size_r = invars["workgroup size r"]
    wg_size = wg_size_c * wg_size_r
    if wg_size > min([x.MAX_WORKGROUP_SIZE for x in matching_devices]):
        return False

    return True

def main():
    """
    Main method.

    Create an experiment and gather data.
    """

    invars = {
        "host": [
            "cec",
            "dhcp-90-060",
            "florence",
            "monza",
            "tim",
            "whz5"
        ],
        "device": [
            ("CPU", 1)
            ("GPU", 1)
            ("GPU", 2)
            ("GPU", 3)
            ("GPU", 4)
        ]
        "border": [
            [ 1,  1,  1,  1],
            [ 5,  5,  5,  5],
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [30, 30, 30, 30],
            [ 1, 10, 30, 30],
            [20, 10, 20, 10]
        ],
        "workgroup size c": [4, 16, 32, 64],
        "workgroup size r": [4, 16, 32, 64]
    }

    # Setup a persistent results store in the current directory.
    lab.setstore(".")

    # Create the experiment by enumerating the possible testcases from
    # the permutations of invars, subject to the constraints provided
    # by "invars_are_legal".
    sample_harness = E14(testcase.cases(invars, type=SimpleBigTest,
                                        contraints=invars_are_legal))

    # Gather data using a static sample size of 100.
    harness.gather(sampler.ProportionalVariance(0.05, min=5, max=30))

    # TODO: Process output.
