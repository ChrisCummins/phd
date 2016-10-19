CxxTargetsObjects = $(addsuffix .o, $(CxxTargets))
CxxTargetsSources = $(addsuffix .cpp, $(CxxTargets))
CxxObjects += $(CxxTargetsObjects)

# Source -> object -> target
BuildTargets += $(CxxTargets)
CxxTargets: $(CxxTargetsObjects)
CxxTargetsObjects: $(CxxTargetsSources)

CleanFiles += $(CxxTargets) $(CxxObjects)

# Compiler flags:

# Inherit optimisation/debug flags from C config:
CxxOptimisationFlags_$(O) = $(COptimisationFlags_$(O))
CxxOptimisationFlags = $(CxxOptimisationFlags_$(O))

CxxDebugFlags_$(D) = $(CDebugFlags_$(D))
CxxDebugFlags = $(CxxDebugFlags_$(D))

CxxFlags = \
	$(CxxOptimisationFlags) \
	$(CxxDebugFlags) \
	-isystem $(build)/llvm/include \
	-std=c++1z \
	-pedantic \
	-Weverything \
	-Wno-c++98-compat \
	-Wno-c++98-compat-pedantic \
	-Wno-documentation \
	-Wno-documentation-unknown-command \
	-Wno-double-promotion \
	-Wno-exit-time-destructors \
	-Wno-float-equal \
	-Wno-global-constructors \
	-Wno-missing-braces \
	-Wno-missing-prototypes \
	-Wno-missing-variable-declarations \
	-Wno-padded \
	-Wno-switch-enum \
	-Wno-unused-parameter \
	-Wno-weak-vtables \
	$(NULL)

# Compile object file from C++ source. Pull in flags from three
# variables: Global, directory-local, and file-local.
#
# To compile object file /path/foo.o from /path/foo.cpp:
#
#     $(CxxFlags)              - Global flags
#     $(/path_CxxFlags)        - Directory-local flags
#     $(/path/foo.o_CxxFlags)  - File-local flags
#
%.o: %.cpp
	$(call cxx-compile-o,$@,$<,\
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags))

cpp: $(CxxTargets)
DocStrings += "cpp: build C++ targets"

.PHONY: print-cxx
print-cxx:
	$(V2)echo $(cxx-compile-o-cmd) $($(PMAKE_INVOC_DIR)_CxxFlags)
DocStrings += "print-cxx: print cxx compiler invocation"
