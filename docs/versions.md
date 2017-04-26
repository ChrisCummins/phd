# CLgen Version Numbering

The installed version is available from the command line:

```sh
$ clgen --version
```

or from the API:

```py
>>> import clgen
>>> clgen.version()
```

CLgen uses a `<major>.<minor>.<micro>[.<dev>]` version numbering scheme. E.g.

* `clgen 0.3.1` means major release 0, minor release 3, micro release 1. 
* `clgen 0.2.10.dev0` means development version 0 of major release 0, minor release 2, micro release 10.

CLgen uses versioned caches, using the path `~/.cache/clgen/<major>.<minor>.x`. This means that caches between different micro and development releases are shared.

## Release Schedule

There is no formal release schedule, but released versions follow the rough guidelines:

* **Development releases** (e.g. `0.1.1` to `0.1.1.dev0`) are not tagged, so there is no "curl one-liner" installation scripts. They may be installed only by cloning the development repository. There may be multiple development releases per micro release.
* **Micro releases** (e.g. `0.1.1` -> `0.1.2`) micro releases (and all other release types) are tagged, so that they may be installed using versioned `install-{deps,cpu,cuda}.sh` scripts. Changes between micro releases may include bug fixes, refactors, and changes to command line and API usage.
* **Minor releases** (e.g. `0.1.3` -> `0.2.0`) any change which breaks compatibility of filesystem caches (e.g. everything under `~/.cache/clgen/<major.<minor>.x`) *must* require a minor release.
* **Major releases** (e.g. `0.10.23` -> `1.0.0`) Who knows? This is still alpha software. Maybe a major release will be the time I sell the company and retire to the Bahamas. Thank you for participating in [our incredible journey](https://ourincrediblejourney.tumblr.com/).
