# My PhD

A self-contained, monolothic repository for (almost) everything I have done while at the University of Edinburgh. Living an #open life.


## Building the code

### Requirements

* Ubuntu Linux or macOS.
* OpenCL.


### Installation

The script `./tools/bootstrap.sh` will probe your system for the required packages, and if any are missing, print the commands necessary to install them. Automatically install them using:

```sh
$ ./tools/bootstrap.sh | bash
```

Test the universe using:

```
$ bazel test //...
```
