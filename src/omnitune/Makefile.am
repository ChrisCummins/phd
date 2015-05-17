SUBDIRS = native

PYTHON = python2

PYTHON_LOG = .test.$(PYTHON).log

check test:
	@echo -n "$(PYTHON): "
	@$(PYTHON) ./setup.py test &> $(PYTHON_LOG) && 		\
		grep -E '^Ran [0-9]+ tests in' $(PYTHON_LOG) ||	\
		cat $(PYTHON_LOG)

install-data-local:
	@sudo $(PYTHON) ./setup.py install

help-local:
	@echo "Makefile options:"
	@echo
	@echo "    make check     Run test suite using python2 and python3"
	@echo "    make install   Install labm8 to system"
