## Generate testcases

Generate testcases using CLSmith:

```sh
$ ./clsmith_mkprogram.py -n 1000
```

Generate testcases using DeepSmith:

```sh
# train and sample model
$ clgen sample model.json sampler.json
# export samples
$ clgen db dump $(clgen --sampler-dir model.json sampler.json)/kernels.json -d /tmp/export
# import samples
$ ./clgen_fetch.py /tmp/export --delete
```

## Run testcases

Collect results for a device:

```sh
$ ./runner.py [--verbose] [--only <testbed_ids>] [--exclude <testbed_ids>] [--batch-size <int>] [--host <hostname>]
```

## Differential test

Prepare results for analysis:

```sh
$ ./set_metas.py
```

Analyze results:

```sh
$ ./analyze.py [--prune]
```

### Reduce interesting testcases

Run automated reductions:

```sh
$ ./run_reductions 0 0 [--clgen|--clsmith]
```

### Prepare interesting testcases for reports

Generate bug reports:

```sh
$ ./report.py
```

Submit bug reports by hand.
