# cldrive - Run arbitrary OpenCL kernels

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
kernel void double_inputs(global int* arr) {
    arr[get_global_id(0)] *= 2;
}
$ cldrive < kernel.cl --devtype=gpu --generator=arange --size 16 -g 16,1,1 -l 4,1,1
arr: [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]
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

See [examples/API.ipynb](/examples/API.ipynb) for a more comprehensive overview of the API.


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
