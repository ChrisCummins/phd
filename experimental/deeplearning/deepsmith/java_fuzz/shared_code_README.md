# Shared code for IBM CAS Java DeepSmith project

This repo contains the subset of Java DeepSmith project code that we have both
been working on. It's probably easiest to submit pull requests here rather than
sending patches back and forth over Slack. We can also use the issue tracker to
keep tabs on bugs and TODOs.


## Java Driver

The file
[deeplearning/deepsmith/harnesses/JavaDriver.java](deeplearning/deepsmith/harnesses/JavaDriver.java)
contains the driver implementation based on Tingda's work. The corresponding
[deeplearning/deepsmith/harnesses/JavaDriverTest.java](deeplearning/deepsmith/harnesses/JavaDriverTest.java)
file contains JUnit tests.


## Java Rewriter

The file
[deeplearning/clgen/preprocessors/JavaRewriter.java](/deeplearning/clgen/preprocessors/JavaRewriter.java)
contains the java rewriter code which Igor has been working on. The tests for
this script are in
[deeplearning/clgen/preprocessors/java_test.py](deeplearning/clgen/preprocessors/java_test.py).
