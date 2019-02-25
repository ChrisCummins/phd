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
```

```sh
$ bazel run //gpu/cldrive -- --src=$PWD/kernel.cl
```


## License

Copyright 2016, 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See 
[LICENSE](/gpu/cldrive/LICENSE) for details.
