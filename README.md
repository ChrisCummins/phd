# cldrive - Run arbitrary OpenCL kernels

## Requirements
* OpenCL
* Python >= 3.6

## Installation

```sh
$ pip install cldrive
```


## Usage

```py
import cldrive

# create an OpenCL environment:
env = cldrive.make_env()

# create a driver for an OpenCL kernel:
driver = cldrive.Driver(env, """
    kernel void double_inputs(global int* data) {
        data[get_global_id(0)] *= 2;
    }""")

# run kernel on some input:
outputs = driver([[0, 1, 2, 3]], gsize=(4, 1, 1), lsize=(1, 1, 1))

print(outputs)  # prints `[[0 2 4 6]]`
```

See the `examples/` directory for Jupyter notebooks with more detailed examples and API documentation.


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
