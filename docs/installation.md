# Building the Development Version

To install the latest development version of CLgen, checkout the development
sources locally using:

```sh
$ git clone https://github.com/ChrisCummins/clgen.git
$ cd clgen
```

Configure and compile CLgen using:

```sh
./configure
make all
```

(Optional) Run the test suite using:

```sh
$ make test
```

Install the python package into the system python using:

```sh
sudo make install
```
