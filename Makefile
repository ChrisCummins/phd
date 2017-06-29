.DEFAULT_GOAL = all

SHELL = /bin/bash

venv_dir := env/python3.6
venv_activate := $(venv_dir)/bin/activate
venv := source $(venv_activate) &&

python_version = 3.6
python = $(venv_dir)/bin/python$(python_version)

all: jupyter clgen cldrive

clgen: $(venv_dir)/bin/clgen

$(venv_dir)/bin/clgen: $(venv_activate)
	$(venv) cd lib/clgen && ./configure -b --with-cuda
	$(venv) cd lib/clgen && make
	$(venv) cd lib/clgen && make test

cldrive: $(venv_dir)/bin/cldrive

$(venv_dir)/bin/cldrive: $(venv_activate)
	$(venv) cd lib/cldrive && make install
	$(venv) cd lib/cldrive && make test

jupyter: $(venv_dir)/bin/jupyter ~/.ipython/kernels/project_b/kernel.json

$(venv_dir)/bin/jupyter: $(venv_activate)
	$(venv) pip install -r requirements.txt

~/.ipython/kernels/project_b/kernel.json: .ipython/kernels/project_b/kernel.json
	mkdir -p ~/.ipython/kernels
	cp -Rv .ipython/kernels/project_b ~/.ipython/kernels

# make Jupyter kernel
.ipython/kernels/project_b/kernel.json: .ipython/kernels/project_b/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(PWD)/$(python)," -i $@

env/python3.6/bin/activate:
	virtualenv -p python3.6 env/python3.6

.PHONY: clean
clean:
	$(venv) cd lib/clgen && make clean
	rm -rfv \
		$(venv_dir)/bin/clgen \
		$(venv_dir)/bin/jupyter \
		$(venv_dir)/bin/cldrive \
		$(NONE)

.PHONY: run
run: all
	$(venv) jupyter-notebook
