# Building the Development Version

To build the latest development version of CLgen, checkout the source
repository locally using:

```sh
$ git clone https://github.com/ChrisCummins/clgen.git clgen-dev
$ cd clgen-dev
```

Configure and compile CLgen using:

```sh
$ ./configure
$ make all
```

Installation - Virtualenv
-------------------------

Create a virtualenv environment in the directory `~/clgen-dev`:

```sh
$ virtualenv --system-site-packages ~/clgen-dev
```

Activate this environment:

```sh
$ source ~/clgen-dev/bin/activate
```

Install CLgen in the virtualenv environment:

```sh
(clgen-dev) $ make install
```

(Optional) Run the test suite:

```sh
(clgen-dev) $ make test
```

When you are done using CLgen, deactivate the virtualenv environment:

```sh
(clgen-dev) $ deactivate
```

To use CLgen later you will need to activate the virtualenv environment again:

```sh
$ source ~/clgen-dev/bin/activate
```


Installation - System-wide
--------------------------

Install CLgen system-wide using:

```sh
$ sudo make install
```

(Optional) Run the test suite:

```sh
$ make test
```