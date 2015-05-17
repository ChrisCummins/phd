#!/usr/bin/env python
"""
In this example script, the user wants to record performance data
for a program compiled at various different optimisations levels using
two different configurations.

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


class RtExperiment(testcase.Harness):
    """
    Experiment harness.
    """
    def setup(self):
        """
        To setup the experiment, change to the correct working directory.
        """
        fs.cd("~/src/rt")

    def teardown(self):
        fs.cdpop()


class RtOlevel(testcase.TestCase):
    """
    Experiment test case.
    """
    TIME_RE = re.compile(".*traces in ([0-9]+.[0-9]+) .*")

    def setup(self):
        """
        Testcase setup.

        To setup a test case, we need to checkout the correct git
        branch, set the optimisation level, and build the program.
        """
        const = self.invars["const"]
        olevel = self.invars["Olevel"]

        # Checkout correct source.
        if const:
            git.checkout("master")
        else:
            git.checkout("no-const")

        # Set Olevel
        system.sed("(^OPTIMISATION_LEVEL = ).*", "\\1{}" % olevel, "Makefile")

        # Build project.
        make.clean()
        make.target("./examples/example1")

    def sample(self):
        """
        Collect a sample.

        Execute the program and graph the runtime from the stdout.
        """
        output = system.check_output("./examples/example1")
        match = re.search(self.TIME_RE, output)
        if match:
            return {"Runtime": float(match.group(1))}


def main():
    """
    Main method.

    Create an experiment, gather the data, and plot the results.
    """
    invars = {
        "Olevel": [
            "-O0",
            "-O1",
            "-O2",
            "-O3",
            "-Os"
        ],
        "const": [True, False]
    }

    # Setup a persistent results store in the current directory.
    lab.setstore(".")

    # Create the experiment by enumerating the possible testcases from
    # the permutations of invars.
    harness = RtExperiment(testcase.cases(invars, type=RtOlevel))

    # Gather data using a static sample size of 100.
    results = harness.gather(sampler.Static(100))

    # Filter the results.
    baseline = results.select("const", False)
    const = results.select("const", True)

    # Plot the results.
    plot.bar("Olevel", "Runtime", baseline, path="basline.png")
    plot.bar("Olevel", "Runtime", const, path="const.png")


if __name__ == "__main__":
    main()
