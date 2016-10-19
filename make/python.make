#
# Python (2 and 3)
#
Python2SetupTestLogs = $(addsuffix /.python2.test.log, \
	$(Python2SetupTestDirs))

Python3SetupTestLogs = $(addsuffix /.python3.test.log, \
	$(Python3SetupTestDirs))

.PHONY: \
	$(Python2SetupTestLogs) \
	$(Python3SetupTestLogs) \
	$(NULL)

$(Python2SetupTestLogs):
	$(call python-setup-test,python2,$(patsubst %/,%,$(dir $@)))

$(Python3SetupTestLogs):
	$(call python-setup-test,python3,$(patsubst %/,%,$(dir $@)))

TestTargets += $(Python2SetupTestLogs) $(Python3SetupTestLogs)

# Clean-up:
Python2CleanDirs = $(sort $(Python2SetupTestDirs))
Python3CleanDirs = $(sort $(Python3SetupTestDirs))

.PHONY: clean-python
clean-python:
	$(V1)$(call python-setup-clean,python2,$(Python2CleanDirs))
	$(V1)$(call python-setup-clean,python3,$(Python3CleanDirs))

CleanTargets += clean-python

CleanFiles += \
	$(Python2SetupTestLogs) \
	$(Python3SetupTestLogs) \
	$(NULL)
