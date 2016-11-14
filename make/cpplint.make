#
# Cpplint
#
CPPLINT := $(root)/make_tools/cpplint.py

CxxLintFilterFlags := \
	build/c++11 \
	build/header_guard \
	build/include_order \
	legal \
	readability/streams \
	readability/todo \
	runtime/references \
	$(NULL)
CxxLintFilters = -$(strip $(call join-with,$(comma)-,\
			$(strip $(CxxLintFilterFlags))))
CxxLintFlags = --root=include --filter=$(CxxLintFilters)

# Deduce:
CppLintTargets = $(addsuffix .cpplint, $(CppLintSources))
BuildTargets += $(CppLintTargets)
CleanFiles += $(CppLintTargets)

%.cpplint: %
	$(call print-task,CPPLINT,$@,$(TaskAux))
	$(call cpplint,$<)
