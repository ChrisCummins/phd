BuildTargets += $(CTargets)

CTargetsObjects = $(addsuffix .o, $(CTargets))
CTargetsSources = $(addsuffix .c, $(CTargets))
CObjects += $(CTargetsObjects)

CTargets: $(CTargetsObjects)
CTargetsObjects: $(CTargetsSources)

CleanFiles += $(CTargets) $(CObjects)

# Compiler flags:
COptimisationFlags_0 = -O0
COptimisationFlags_1 = -O2
COptimisationFlags = $(COptimisationFlags_$(O))

# Debug flags:
CDebugFlags_1 = -g
CDebugFlags = $(CDebugFlags_$(D))

CFlags = \
	$(COptimisationFlags) \
	$(CDebugFlags) \
	-std=c11 \
	-pedantic \
	-Weverything \
	-Wno-bad-function-cast \
	-Wno-double-promotion \
	-Wno-missing-prototypes \
	-Wno-missing-variable-declarations \
	-Wno-unused-parameter \
	$(NULL)

%.o: %.c
	$(call c-compile-o,$@,$<,\
		$($(patsubst %/,%,$@)_CFlags) \
		$($(patsubst %/,%,$(dir $@))_CFlags))

c: $(CTargets)
DocStrings += "c: build C targets"

.PHONY: print-cc
print-cc:
	$(V2)echo $(c-compile-o-cmd) $($(PMAKE_INVOC_DIR)_CxxFlags)
DocStrings += "print-cc: print cc compiler invocation"
