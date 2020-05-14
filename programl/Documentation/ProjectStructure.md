# Project Structure

This project is divided into the following top level packages:

* `cmd/` Commandline tools. All executable binaries are stored here.
* `Documentation/` Additional documentation.
* `graph/` Libraries for creating and manipulating the ProGraML graph
  representation.
* `ir/` Support for specific compiler IRs, e.g. LLVM.
* `proto/` Protocol message definitions. These define the core data
  structures for this project. This is a good starting point for
  understanding the code.
* `task/` Experimental tasks. This contains the code for producing the
  results of our published experiments.
* `test/` Testing utilities and helpers.
* `util/` Utility libraries.
