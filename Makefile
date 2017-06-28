SHELL = /bin/bash

venv_dir := env/python3.6
venv_activate := $(venv_dir)/bin/activate
venv := source $(venv_activate) &&

all: jupyter

clgen: $(venv_dir)/bin/clgen

$(venv_dir)/bin/clgen: $(venv_activate)
	$(venv) cd lib/clgen && ./configure -b --with-cuda
	$(venv) cd lib/clgen && make
	$(venv) cd lib/clgen && make test

jupyter: $(venv_dir)/bin/jupyter

$(venv_dir)/bin/jupyter: clgen
	$(venv) pip install -r reuirements.txt

env/python3.6/bin/activate:
	virtualenv -p python3.6 env/python3.6

.PHONY: clean
clean:
	$(venv) cd lib/clgen && make clean
	rm -rfv $(venv_dir)/bin/clgen
