# cldrive - Run arbitrary OpenCL kernels

<a href="https://badge.fury.io/py/cldrive">
  <img src="https://img.shields.io/pypi/v/cldrive.svg?colorB=green&style=flat">
</a>
<a href="https://travis-ci.org/ChrisCummins/cldrive" target="_blank">
  <img src="https://img.shields.io/travis/ChrisCummins/cldrive/master.svg?style=flat">
</a>
<a href="http://chriscummins.cc/cldrive" target="_blank">
  <img src="https://img.shields.io/badge/docs-latest-green.svg?style=flat">
</a>
<a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
  <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
</a>

## Requirements
* OpenCL
* Python >= 3.6

## Installation

```sh
$ pip install cldrive
```


## Usage

From the command line:
```sh
$ cat kernel.cl
kernel void my_kernel(global int* a, global int* b) {
    int tid = get_global_id(0);
    a[tid] += 1;
    b[tid] = a[tid] * 2;
}
$ cldrive < kernel.cl --devtype=gpu --generator=arange --size 12 -g 12,1,1 -l 4,1,1
a: [ 1  2  3  4  5  6  7  8  9 10 11 12]
b: [ 2  4  6  8 10 12 14 16 18 20 22 24]
```

From Python:

```py
import cldrive

# our OpenCL kernel to run:
src = """
    kernel void double_inputs(global int* data) {
        data[get_global_id(0)] *= 2;
    }
"""

# the data to run it on:
inputs = [[0, 1, 2, 3]]

# create an OpenCL environment for the first available GPU:
env = cldrive.make_env(devtype="gpu")

# run kernel on the input:
outputs = cldrive.drive(env, src, inputs, gsize=(4, 1, 1), lsize=(1, 1, 1))

print(outputs)  # prints `[[0 2 4 6]]`
```

See the [API Documentation](http://chriscummins.cc/cldrive) for more details.


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
