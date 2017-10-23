# Building the Development Version

To build the latest development version of CLgen, checkout the latest
development sources using:

```sh
$ git clone git@github.com:ChrisCummins/clgen.git clgen-dev
$ cd clgen-dev
```

Install the build requirements using:

```
$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-deps.sh | bash
```

Configure CLgen using:

```sh
$ ./configure
```

Installation - Virtualenv
-------------------------

Create a virtualenv environment in the directory `~/clgen-dev`:

```sh
$ virtualenv -p python3 ~/clgen-dev
```

Activate this environment:

```sh
$ source ~/clgen-dev/bin/activate
```

Install CLgen in the virtualenv environment:

```sh
(clgen-dev)$ make all
```

(Optional) Run the test suite:

```sh
(clgen-dev)$ make test
```

(Optional) Install Python development tools:

```sh
(clgen-dev)$ pip install -r make/requirements/requirements.dev.txt
```

When you are done using CLgen, deactivate the virtualenv environment:

```sh
(clgen-dev)$ deactivate
```

To use CLgen later you will need to activate the virtualenv environment again:

```sh
$ source ~/clgen-dev/bin/activate
```


Installation - System-wide
--------------------------

Install CLgen system-wide using:

```sh
$ sudo -H make all
```

(Optional) Run the test suite:

```sh
$ make test
```

(Optional) Install Python development tools:

```sh
$ pip install -r make/requirements/requirements.dev.txt
```
