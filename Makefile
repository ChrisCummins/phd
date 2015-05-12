PYTHON3 = python3
PYTHON2 = python2

PYTHON3_LOG = .test.python3.log
PYTHON2_LOG = .test.python2.log

check test:
	@echo -n "Python3: "
	@$(PYTHON3) ./setup.py test &> $(PYTHON3_LOG) && 		\
		grep -E '^Ran [0-9]+ tests in' $(PYTHON3_LOG) || 	\
		cat $(PYTHON3_LOG)

	@echo -n "Python2: "
	@$(PYTHON2) ./setup.py test &> $(PYTHON2_LOG) && 		\
		grep -E '^Ran [0-9]+ tests in' $(PYTHON2_LOG) ||	\
		cat $(PYTHON2_LOG)

install:
	@sudo $(PYTHON3) ./setup.py install
	@sudo $(PYTHON2) ./setup.py install

help:
	@echo "Makefile options:"
	@echo
	@echo "    make check     Run test suite using python2 and python3"
	@echo "    make install   Install labm8 to system"
