Generate CLSmith programs:

```sh
$ ./clsmith_mkprogram.py -n 1000
```

Generate CLgen programs:

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
$ ./run-programs.py 0 0 [--clgen|--clsmith] [--t6h] [--opt|--no-opt]
```

Prepare results for analysis:

```sh
$ ./set_metas.sh
```

Analyze results:

```sh
$ ./analyze.py [--clgen|--clsmith] [--prune]
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
