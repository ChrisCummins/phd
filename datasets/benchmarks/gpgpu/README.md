# GPGPU Benchmarks

This package contains seven popular GPGPU benchmark suites. The benchmarks have
been modified to use the [//gpu/libcecl](/gpu/libcecl) library. This allows
the OpenCL kernels to be executed on specific hardware, and for verbose timing
information to be provided when executing OpenCL kernels.

### References

Data derived from this set of benchmarks have been used in these publications:

* C. Cummins, P. Petoumenos, W. Zang, and H. Leather,
  "[Synthesizing Benchmarks for Predictive Modeling](/docs/2017_02_cgo),"
  in CGO, 2017.
```
@inproceedings{cummins2017a,
  title={Synthesizing Benchmarks for Predictive Modeling},
  author={Cummins, Chris and Petoumenos, Pavlos and Wang, Zheng and Leather, Hugh},
  booktitle={CGO},
  year={2017},
  organization={IEEE}
}
```

* C. Cummins, P. Petoumenos, Z. Wang, and H. Leather,
  "[End-to-end Deep Learning of Optimization Heuristics](/docs/2017_09_pact),"
  in PACT, 2017.
```
@inproceedings{cummins2017b,
  title={End-to-end Deep Learning of Optimization Heuristics},
  author={Cummins, Chris and Petoumenos, Pavlos and Wang, Zheng and Leather, Hugh},
  booktitle={PACT},
  year={2017},
  organization={ACM}
}
```
