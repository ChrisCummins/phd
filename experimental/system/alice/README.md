# Alice: A Little Command Executor

Alice is a tool for keeping track of commands that I run, with a poorly
retrofitted acronym. The idea is that it keeps track of the execution of bazel 
targets in this repo, so that I remember/recreate them. It does this using a
ledger; not a fancy distributed ledger like all the new kids are using, just a
good old fashioned list of commands, their output, and some other useful data.
Basically it's a jazzed up `.bash_history`.

### The life time of an execution

```sh
$ alice exec //src:experiment --data_root /tmp/experiment
```

1. Create and fill a `LedgerEntry` proto with the pre-execution stats.
1. Submit the `LedgerEntry` to the ledger service and obtain an entry ID.
1. Report the ID to stdout.
1. Build the requested bazel target.
1. Determine the path of the built binary.
1. Execute the binary with the requested arguments.
1. Spool command output to stdout until command completes.
1. Record end-of-execution stats.
1. Submit the final `LedgerEntry` to the ledger service.
1. Exit with the command's return code.


### Other commands

```sh
$ alice status [--host=<hostname>]
```

Print a list of currently-active jobs.

```sh
$ alice log [--host=<hostname>] [--with_active] 
```


### Possible Future Work

* Job quotas, CPU pinning, GPU pinning, process priorities.
* Schedule jobs to execute upon completion of an existing ledger ID. 
