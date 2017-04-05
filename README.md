# cldrive - Run arbitrary OpenCL kernels

## Requirements
* OpenCL
* Python >= 3.6

## Installation

```sh
$ pip install cldrive
```

## Usage

From command line:

```sh
$ cat kernel.cl
__kernel void A(__global int* data) {
    int tid = get_global_id(0);
    data[tid] *= 2.0
}
$ cldrive kernel.cl -g=4,1,1 -l=1,1,1 -i="seq"
0, 2, 4, 6, 8
```

From Python:

```py
import cldrive

kernel = """\
__kernel void A(__global int* data) {
    int tid = get_global_id(0);
    data[tid] *= 2.0
}
"""

outputs = cldrive.run_kernel(kernel, inputs=cldrive.Inputs.SEQ,
                             gsize=cldrive.NDRange(4,1,1),
                             lsize=cldrive.NDRange(1,1,1))
print(outputs)  # output: [0, 2, 4, 6, 8]
```


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
