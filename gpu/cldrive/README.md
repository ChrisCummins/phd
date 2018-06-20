# cldrive - Run arbitrary OpenCL kernels

<a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
  <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
</a>


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
from gpu import cldrive

# The OpenCL kernel to run.
src = """
  kernel void double_inputs(global int* data) {
    data[get_global_id(0)] *= 2;
  }
"""

# The data to run it on.
inputs = [[0, 1, 2, 3]]

# Create an OpenCL environment for the first available GPU.
env = cldrive.make_env(devtype="gpu")

# Run kernel on the input.
outputs = cldrive.drive(env, src, inputs, gsize=(4, 1, 1), lsize=(1, 1, 1))

print(outputs)  # prints `[[0 2 4 6]]`
```


## License

Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See 
[LICENSE](/gpu/cldrive/LICENSE) for details.
