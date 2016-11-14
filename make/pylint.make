#
# Pylint - pep8
#
PYLINT := pep8

PyLintFlags := \
	--show-source \
	--ignore=E231,E701 \
	$(NULL)
PyLintTargets = $(addsuffix .pylint, $(PyLintSources))
BuildTargets += $(PyLintTargets)
CleanFiles += $(PyLintTargets)

%.pylint: %
	$(call print-task,PYLINT,$@,$(TaskAux))
	$(call pylint,$@,$<)


lint: $(CppLintTargets) $(PyLintTargets)
DocStrings += "lint: build lint files"
