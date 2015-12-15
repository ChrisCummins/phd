# I can haz pythons?
PYTHONS := python2 python3

.PHONY: check test install help

check test:
	@for python in $(PYTHONS); do						\
		echo -n "$$python: ";						\
		$$python ./setup.py test &>.test.$$python.log &&		\
			grep -E '^Ran [0-9]+ tests in' .test.$$python.log || 	\
		cat .test.$$python.log;						\
	done

# I can haz root permissions si?
install:
	@for python in $(PYTHONS); do						\
		echo -n "$$python install: ";					\
			sudo $$python ./setup.py install &>.install.$$python.log && \
			echo "ok" || cat .install.$$python.log;			\
	done

help:
	@echo "Makefile commands:"
	@echo
	@echo "    make check     Run test suites"
	@echo "    make install   Install labm8 to system (requires sudo)"
	@echo
	@echo "Makefile options:"
	@echo
	@echo "    Python versions: [$(PYTHONS)]."
