* Get experimental data using benchmarks `./automate` script.
* Move logs into `data/logs/<run>/<suite/<log>`.
* Generate device values with `cecl2features -d data/logs`.
* Move kernels into `data/kernels/<kernel>`.
* Make a list of kernel files `data/kernels/file-list.txt`.
* Generate kernels with `cd data/kernels && smith-features $(cat
  file-list.txt | grep -v '#') --fatal-errors --shim >features.csv`.
* Make features data with `./mkfeatures data/kernels data/logs`.
* Move device CSV files into `data/logs`, e.ge. `cp -v data/logs/amd.csv .`.
* Make training data with:

```
./mktraining data/intel.csv data/amd.csv > data/platform-a.csv
./mktraining data/intel.csv data/nvidia.csv > data/platform-b.csv
```

* Eval training data with:

```
cgo13 data/platform-a.csv 2>/dev/null | column -t -s','
```
