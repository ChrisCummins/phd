Generate CLSmith programs:

```sh
$ ./clsmith_mkprogram.py -n 1000
```

Generate DeepSmith programs:

```sh
# train and sample model
$ clgen sample model.json sampler.json
# export samples
$ clgen db dump $(clgen --sampler-dir model.json sampler.json)/kernels.json -d /tmp/export
# import samples
$ ./clgen_fetch.py /tmp/export --delete
```

Collect results for a device:

```sh
$ ./runner.py [--verbose] [--only <testbed_ids>] [--exclude <testbed_ids>] [--batch-size <int>] [--host <hostname>]
```

Prepare results for analysis:

```sh
$ ./set_metas.py
```

Analyze results:

```sh
$ ./analyze.py [--prune]
```

Run automated reductions:

```sh
$ ./run_reductions 0 0 [--clgen|--clsmith]
```

Generate bug reports:

```sh
$ ./report.py
```

Submit bug reports by hand.

Record submitted bug report:

```sh
$ ./submit.py <testbed-id> <testcase_url> --reported <report_url> [--fixed]
```
