# path to python3
PYTHON := python3
# path to virtualenv
VIRTUALENV := virtualenv


# Rules to create virtualenv:

# name of virtualenv (you can leave this)
VIRTUALENV_BASE := env

# create virtualenv
virtualenv: $(VIRTUALENV_BASE)/bin/activate
$(VIRTUALENV_BASE)/bin/activate:
	$(VIRTUALENV) -p $(PYTHON) $(VIRTUALENV_BASE)

# source virtualenv
env := source $(VIRTUALENV_BASE)/bin/activate &&


# Targets:

# run tests
test: virtualenv
	$(env)python ./setup.py test
