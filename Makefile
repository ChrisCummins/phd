# The default goal is...
.DEFAULT_GOAL = all

# Use V=1 argument for verbose builds
QUIET_  = @
QUIET   = $(QUIET_$(V))

# Assume no out-of-tree builds:
root := $(PWD)

SHELL := /bin/bash
NPROC := 4

comma := ,
space :=
space +=

#
# Configuration
#
AWK := awk
EGREP := egrep
GREP := grep
MAKEFLAGS := -j$(NPROC)
PYTHON2 := python2
PYTHON3 := python3
RM := rm -fv
SED := sed

# Targets:
AutotexTargets =
BuildTargets =
CleanFiles =
CleanTargets =
CTargets =
CxxTargets =
CppLintSources =
DistcleanFiles =
DistcleanTargets =
DontLint =
InstallTargets =
Python2SetupInstallDirs =
Python2SetupTestDirs =
Python3SetupInstallDirs =
Python3SetupTestDirs =
TestTargets =


########################################################################
#                             Functions

# Joins elements of a list
#
# Arguments:
#   $1 (str)   Separator
#   $2 (str[]) List
define join-with
	$(subst $(space),$1,$(strip $2))
endef

# Compile C sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C sources
#   $3 (str[]) Flags for compiler
define c-compile-o
	@echo '  CC       $1'
	$(QUIET)$(CC) $(CFlags) $3 $2 -c -o $1
endef

# Compile C++ sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C++ sources
#   $3 (str[]) Flags for compiler
define cxx-compile-o
	@echo '  CXX      $1'
	$(QUIET)$(CXX) $(CxxFlags) $3 $2 -c -o $1
	$(call clang-tidy,$2,$3)
	$(call cpplint,$2)
endef

# Link object files to executable
#
# Arguments:
#   $1 (str)   Executable
#   $2 (str[]) Object files
#   $3 (str[]) Flags for linker
define o-link
	@echo '  LD       $1'
	$(QUIET)$(LD) $(CxxFlags) $(LdFlags) $3 $2 -o $1
endef

# Run cpplint on input, generating a .lint file.
#
# Arguments:
#   $1 (str) C++ source/header
define cpplint
	$(QUIET)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		$(CPPLINT) $(CxxLintFlags) $1 2>&1 \
			| grep -v '^Done processing\|^Total errors found: ' \
			| tee $1.lint; \
		fi
endef

# Run clang-tidy on input.
#
# Arguments:
#  $1 (str) source file
#  $2 (str[]) Compilation flags
define clang-tidy
	$(QUIET)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		$(CLANGTIDY) $1 -- $(CxxFlags) $2; \
	fi
endef

# Run python setup.py test
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-test
	@echo '  TEST    $(strip $1) $(strip $2)'
	$(QUIET)cd $2 && $(strip $1) ./setup.py test \
		&> $2/.$(strip $1).test.log \
		&& $(GREP) -E '^Ran [0-9]+ tests in' \
		$2/.$(strip $1).test.log \
		|| sed -n -e '/ \.\.\. /,$${p}' $2/.$(strip $1).test.log | \
		grep -v '... ok'
endef

# Run python setup.py install
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-install
	@echo '  INSTALL $2'
	$(QUIET)cd $2 && $(strip $1) ./setup.py install \
		&> $2/.$(strip $1).install.log \
		|| cat $2/.$(strip $1).install.log
endef

# Run python setup.py clean
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-clean
	for dir in $2; do \
		cd $$dir && $1 ./setup.py clean >/dev/null; \
	done
endef


########################################################################
#                             Targets

#
# docs/
#
AutotexTargets += \
	$(root)/docs/2015-01-msc-thesis/thesis.pdf \
	$(root)/docs/2015-02-progression-review/document.pdf \
	$(root)/docs/2016-01-adapt/adapt.pdf \
	$(root)/docs/2016-02-hlpgpu/hlpgpu.pdf \
	$(root)/docs/wip-outline/outline.pdf \
	$(NULL)


#
# extern/
#
extern := $(root)/extern

#
# extern/benchmark
#
GoogleBenchmark = $(extern)/benchmark/build/src/libbenchmark.a
GoogleBenchmark_CxxFlags = -I$(extern)/benchmark/include
GoogleBenchmark_LdFlags = -L$(extern)/benchmark/build/src -lbenchmark

$(GoogleBenchmark):
	@echo '  BUILD    $@'
	$(QUIET)mkdir -pv $(extern)/benchmark/build
	$(QUIET)cd $(extern)/benchmark/build \
		&& cmake .. \
		&& $(MAKE)

.PHONY: distclean-googlebenchmark
distclean-googlebenchmark:
	$(QUIET)$(RM) -r $(extern)/benchmark/build

DistcleanTargets += distclean-googlebenchmark


#
# extern/googletest
#
GoogleTest = $(extern)/googletest-build/libgtest.a
GoogleTest_CxxFlags = -I$(extern)/googletest/googletest/include
GoogleTest_LdFlags = -L$(extern)/googletest-build -lgtest

$(GoogleTest):
	@echo '  BUILD    $@'
	$(QUIET)mkdir -pv $(extern)/googletest-build
	$(QUIET)cd $(extern)/googletest-build \
		&& cmake ../googletest/googletest \
		&& $(MAKE)

.PHONY: distclean-googletest
distclean-googletest:
	$(QUIET)$(RM) -r $(extern)/googletest-build

DistcleanTargets += distclean-googletest


#
# lab/
#
lab := $(root)/lab

#
# lab/stl/
#
StlComponents = \
	algorithm \
	array \
	forward_list \
	map \
	unordered_map \
	vector \
	$(NULL)

StlHeaders = $(addprefix $(lab)/stl/include/ustl/,$(StlComponents))
Stl_CxxFlags = -I$(lab)/stl/include

# Stl unit tests:
StlTestsSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/tests/,$(StlComponents)))
StlTestsObjects = $(patsubst %.cpp,%.o,$(StlTestsSources))
$(StlTestsObjects): $(StlTestsSources) $(StlHeaders) $(GoogleBenchmark)
$(lab)/stl/tests/tests: $(StlTestsObjects)

CxxTargets += $(lab)/stl/tests/tests
$(lab)/stl/tests_CxxFlags = $(Stl_CxxFlags) $(GoogleTest_CxxFlags)
$(lab)/stl/tests_LdFlags = $(GoogleTest_LdFlags)

# Stl benchmarks:
StlBenchmarksSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/benchmarks/,$(StlComponents)))
StlBenchmarksObjects = $(patsubst %.cpp,%.o,$(StlBenchmarksSources))
$(StlBenchmarksObjects): $(StlBenchmarksSources) $(StlHeaders) $(GoogleBenchmark)
$(lab)/stl/benchmarks/benchmarks: $(StlBenchmarksObjects)

CxxTargets += $(lab)/stl/benchmarks/benchmarks
$(lab)/stl/benchmarks_CxxFlags = $(Stl_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(lab)/stl/benchmarks_LdFlags = $(GoogleBenchmark_LdFlags)

#
# learn/
#
learn := $(root)/learn


#
# learn/atc++/
#
CxxTargets += \
	$(learn)/atc++/benchmark-argument-type \
	$(learn)/atc++/constructors \
	$(learn)/atc++/myvector \
	$(learn)/atc++/strings \
	$(learn)/atc++/user-input \
	$(NULL)

$(learn)/atc++/benchmark-argument-type.o_CxxFlags = $(GoogleBenchmark_CxxFlags)
$(learn)/atc++/benchmark-argument-type_LdFlags = $(GoogleBenchmark_LdFlags)
$(learn)/atc++/benchmark-argument-type.o: $(GoogleBenchmark)

$(learn)/atc++/constructors.o_CxxFlags = $(GoogleTest_CxxFlags)
$(learn)/atc++/constructors_LdFlags = $(GoogleTest_LdFlags)
$(learn)/atc++/constructors.o: $(GoogleTest)


#
# learn/challenges/
#
CxxTargets += \
	$(learn)/challenges/001-int-average \
	$(learn)/challenges/006-gray-code \
	$(learn)/challenges/008-linked-list \
	$(NULL)

$(learn)/challenges_CxxFlags = \
	$(GoogleBenchmark_CxxFlags) $(GoogleTest_CxxFlags)
$(learn)/challenges_LdFlags = \
	$(GoogleBenchmark_LdFlags) $(GoogleTest_LdFlags)
$(learn)/challenges/01-int-average.o: $(GoogleBenchmark) $(GoogleTest)


#
# learn/ctci/
#
CtCiTargets = \
	$(learn)/ctci/0101-unique-chars-in-string \
	$(learn)/ctci/0102-reverse-string \
	$(learn)/ctci/0103-permutations \
	$(learn)/ctci/0104-escape-string \
	$(learn)/ctci/0105-string-compression \
	$(learn)/ctci/0107-matrix-zero \
	$(learn)/ctci/0108-string-rotation \
	$(learn)/ctci/0201-list-remove-dups \
	$(learn)/ctci/0202-linked-list-k-last \
	$(learn)/ctci/0302-stack-min \
	$(learn)/ctci/0402-directed-graph-routefinder \
	$(learn)/ctci/0502-binary-double \
	$(learn)/ctci/1101-merge-arrays \
	$(learn)/ctci/1102-sort-anagrams \
	$(learn)/ctci/1301-last-k-lines \
	$(learn)/ctci/1307-tree-copy \
	$(learn)/ctci/1701-num-swap \
	$(learn)/ctci/1702-tic-tac-toe \
	$(NULL)

CxxTargets += $(CtCiTargets)

$(learn)/ctci_CxxFlags = $(GoogleBenchmark_CxxFlags) $(GoogleTest_CxxFlags)
$(learn)/ctci_LdFlags = $(GoogleBenchmark_LdFlags) $(GoogleTest_LdFlags)
CtCiObjects = $(addsuffix .o,$(CtCiTargets))
$(CtCiObjects): $(GoogleBenchmark) $(GoogleTest)


#
# learn/expert_c/
#
CTargets += \
	$(learn)/expert_c/cdecl \
	$(learn)/expert_c/computer_dating \
	$(NULL)


#
# playground/
#

#
# playground/rt/
#
RayTracerDir = $(root)/playground/rt
RayTracerLib = $(RayTracerDir)/src/librt.so

CxxTargets += \
	$(RayTracerDir)/examples/example1 \
	$(RayTracerDir)/examples/example2 \
	$(NULL)

$(RayTracerDir)/examples/example1: $(RayTracerLib)

$(RayTracerDir)/examples/example2: \
		$(RayTracerDir)/examples/example2.o \
		$(RayTracerLib) \
		$(NULL)

$(RayTracerDir)/examples/example2.cpp: \
		$(RayTracerDir)/examples/example2.rt \
		$(RayTracerDir)/scripts/mkscene.py
	@echo "  MKSCENE  $@"
	$(QUIET)$(RayTracerDir)/scripts/mkscene.py $< $@ >/dev/null

# Don't run linter on generated file:
DontLint += $(RayTracerDir)/examples/example2.cpp

RayTracerHeaders = \
	$(RayTracerDir)/include/rt/camera.h \
	$(RayTracerDir)/include/rt/graphics.h \
	$(RayTracerDir)/include/rt/image.h \
	$(RayTracerDir)/include/rt/lights.h \
	$(RayTracerDir)/include/rt/math.h \
	$(RayTracerDir)/include/rt/profiling.h \
	$(RayTracerDir)/include/rt/random.h \
	$(RayTracerDir)/include/rt/renderer.h \
	$(RayTracerDir)/include/rt/rt.h \
	$(RayTracerDir)/include/rt/scene.h \
	$(NULL)

CppLintSources += $(RayTracerHeaders)

RayTracerSources = $(wildcard $(RayTracerDir)/src/*.cpp)
RayTracerObjects = $(patsubst %.cpp, %.o, $(RayTracerSources))

$(RayTracerObjects): $(RayTracerHeaders) $(toolchain)

# Project specific flags:
RayTracerCxxFlags = -I$(RayTracerDir)/include
$(RayTracerDir)/src_CxxFlags = $(RayTracerCxxFlags)
$(RayTracerDir)/examples_CxxFlags = $(RayTracerCxxFlags)
$(RayTracerDir)/examples_LdFlags = -ltbb -lrt -L$(dir $(RayTracerLib))

# Link library:
$(RayTracerLib): $(RayTracerObjects)
	$(call o-link,$@,$?,-fPIC -shared)

CleanFiles += $(RayTracerObjects) $(RayTracerLib)

#
# playground/sc/
#
CxxTargets += $(root)/playground/sc/sc


#
# src/
#

# src/labm8
Python2SetupTestDirs += $(root)/src/labm8
Python2SetupInstallDirs += $(root)/src/labm8
Python3SetupTestDirs += $(root)/src/labm8
Python3SetupInstallDirs += $(root)/src/labm8

# src/omnitune
Python2SetupTestDirs += $(root)/src/omnitune
Python2SetupInstallDirs += $(root)/src/omnitune

#
# thesis/
#
AutotexTargets += $(root)/thesis/thesis.pdf


########################################################################
#                         Build rules

#
# C
#
CC := $(root)/tools/llvm/build/bin/clang

BuildTargets += $(CTargets)

CObjects = $(addsuffix .o, $(CTargets))
CSources = $(addsuffix .c, $(CTargets))

CTargets: $(CObjects)
CObjects: $(CSources)

CleanFiles += $(CTargets) $(CObjects)

CFlags = \
	-O2 \
	-std=c11 \
	-pedantic \
	-Wall \
	-Wextra \
	-Wcast-align \
	-Wcast-qual \
	-Wctor-dtor-privacy \
	-Wdisabled-optimization \
	-Wformat=2 \
	-Wframe-larger-than=1024 \
	-Winit-self \
	-Winline \
	-Wlarger-than=2048 \
	-Wmissing-declarations \
	-Wmissing-include-dirs \
	-Wno-div-by-zero \
	-Wno-main \
	-Wno-missing-braces \
	-Wno-unused-parameter \
	-Wold-style-cast \
	-Woverloaded-virtual \
	-Wpadded \
	-Wredundant-decls \
	-Wshadow \
	-Wsign-conversion \
	-Wsign-promo \
	-Wstrict-overflow=5 \
	-Wswitch-default \
	-Wundef \
	-Wwrite-strings \
	$(NULL)

%.o: %.c
	$(call c-compile-o,$@,$<,\
		$($(patsubst %/,%,$@)_CFlags) \
		$($(patsubst %/,%,$(dir $@))_CFlags))


#
# C++
#
CXX := $(root)/tools/llvm/build/bin/clang++
CLANGTIDY := $(root)/tools/llvm/build/bin/clang-tidy

CxxObjects = $(addsuffix .o, $(CxxTargets))
CxxSources = $(addsuffix .cpp, $(CxxTargets))

# Source -> object -> target
BuildTargets += $(CxxTargets)
CxxTargets: $(CxxObjects)
CxxObjects: $(CxxSources)

CleanFiles += $(CxxTargets) $(CxxObjects)

# Compiler flags:
CxxFlags = \
	-O2 \
	-std=c++14 \
	-stdlib=libc++ \
	-pedantic \
	-Wall \
	-Wextra \
	-Wcast-align \
	-Wcast-qual \
	-Wctor-dtor-privacy \
	-Wdisabled-optimization \
	-Wformat=2 \
	-Wframe-larger-than=2048 \
	-Winit-self \
	-Winline \
	-Wlarger-than=2048 \
	-Wmissing-declarations \
	-Wmissing-include-dirs \
	-Wno-div-by-zero \
	-Wno-main \
	-Wno-missing-braces \
	-Wno-unused-parameter \
	-Wold-style-cast \
	-Woverloaded-virtual \
	-Wpadded \
	-Wredundant-decls \
	-Wshadow \
	-Wsign-conversion \
	-Wsign-promo \
	-Wstrict-overflow=5 \
	-Wswitch-default \
	-Wundef \
	-Wwrite-strings \
	$(NULL)

%.o: %.cpp
	$(call cxx-compile-o,$@,$<,\
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags))

#
# Cpplint
#
CPPLINT := $(root)/tools/cpplint.py

CxxLintFilterFlags = \
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
CppLintTargets = $(addsuffix .lint, $(CppLintSources))
BuildTargets += $(CppLintTargets)
CleanFiles += $(CppLintTargets)

%.h.lint: %.h
	@echo '  LINT     $@'
	$(call cpplint,$<)

#
# Linker
#
LD := $(CXX)

LdFlags =

%: %.o
	$(call o-link,$@,$(filter %.o,$^),\
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags) \
		$($(patsubst %/,%,$@)_LdFlags) \
		$($(patsubst %/,%,$(dir $@))_LdFlags))


#
# LaTeX
#
AUTOTEX := $(root)/tools/autotex.sh

BuildTargets += $(AutotexTargets)

AutotexDirs = $(dir $(AutotexTargets))
AutotexDepFiles = $(addsuffix .autotex.deps, $(AutotexDirs))
AutotexLogFiles = $(addsuffix .autotex.log, $(AutotexDirs))

.PHONY: $(AutotexTargets)
$(AutotexTargets):
	@$(AUTOTEX) make $(patsubst %.pdf,%,$@)
# Autotex does it's own dependency analysis, so always run it:

# File extensions to remove in LaTeX build directories:
LatexBuildfileExtensions = \
	-blx.bib \
	.acn \
	.acr \
	.alg \
	.aux \
	.bbl \
	.bcf \
	.blg \
	.dvi \
	.fdb_latexmk \
	.glg \
	.glo \
	.gls \
	.idx \
	.ilg \
	.ind \
	.ist \
	.lof \
	.log \
	.lol \
	.lot \
	.maf \
	.mtc \
	.mtc0 \
	.nav \
	.nlo \
	.out \
	.pdfsync \
	.ps \
	.run.xml \
	.snm \
	.synctex.gz \
	.tdo \
	.toc \
	.vrb \
	.xdy \
	$(NULL)

LatexBuildDirs = $(AutotexDirs)

# Discover files to remove using the shell's `find' tool:
LatexCleanFiles = $(shell find $(LatexBuildDirs) \
	-name '*$(call join-with,' -o -name '*, $(LatexBuildfileExtensions))')

CleanFiles += \
	$(AutotexTargets) \
	$(AutotexDepFiles) \
	$(AutotexLogFiles) \
	$(LatexCleanFiles) \
	$(NULL)


#
# Python
#
Python2SetupTestLogs = $(addsuffix /.python2.test.log, \
	$(Python2SetupTestDirs))

Python2SetupInstallLogs = $(addsuffix /.python2.install.log, \
	$(Python2SetupInstallDirs))

Python3SetupTestLogs = $(addsuffix /.python3.test.log, \
	$(Python3SetupTestDirs))

Python3SetupInstallLogs = $(addsuffix /.python3.install.log, \
	$(Python3SetupInstallDirs))

.PHONY: \
	$(Python2SetupInstallLogs) \
	$(Python2SetupTestLogs) \
	$(Python3SetupInstallLogs) \
	$(Python3SetupTestLogs) \
	$(NULL)

$(Python2SetupTestLogs):
	$(call python-setup-test,$(PYTHON2),$(patsubst %/,%,$(dir $@)))

$(Python2SetupInstallLogs):
	$(call python-setup-install,$(PYTHON2),$(patsubst %/,%,$(dir $@)))

$(Python3SetupTestLogs):
	$(call python-setup-test, $(PYTHON3),$(patsubst %/,%,$(dir $@)))

$(Python3SetupInstallLogs):
	$(call python-setup-install,$(PYTHON3),$(patsubst %/,%,$(dir $@)))

TestTargets += $(Python2SetupTestLogs) $(Python3SetupTestLogs)
InstallTargets += $(Python2SetupInstallLogs) $(Python3SetupInstalLogs)

# Clean-up:
Python2CleanDirs = $(sort $(Python2SetupTestDirs) $(Python2SetupInstallDirs))
Python3CleanDirs = $(sort $(Python3SetupTestDirs) $(Python3SetupInstallDirs))

.PHONY: clean-python
clean-python:
	$(QUIET)$(call python-setup-clean,$(PYTHON2),$(Python2CleanDirs))
	$(QUIET)$(call python-setup-clean,$(PYTHON3),$(Python3CleanDirs))

CleanTargets += clean-python

CleanFiles += \
	$(Python2SetupInstallLogs) \
	$(Python2SetupTestLogs) \
	$(Python3SetupInstallLogs) \
	$(Python3SetupTestLogs) \
	$(NULL)


#
# Testing
#
test: $(TestTargets)


#
# Install
#
install: $(InstallTargets)


#
# Bootstrapping
#
toolchain := $(root)/.bootstrapped

LlvmSrc := $(root)/tools/llvm
LlvmBuild := $(root)/tools/llvm/build

# Compilers depend on boostrapping:
$(CC): $(toolchain)
$(CTargets) $(CObjects): $(CC)

$(CXX): $(toolchain)
$(CxxTargets): $(CXX)
$(CxxObjects): $(CXX)

$(toolchain):
	@echo "Bootstrapping! Go enjoy a coffee, this will take a while."
	$(QUIET)mkdir -vp $(LlvmBuild)
	$(QUIET)cd $(LlvmBuild) \
		&& cmake $(LlvmSrc) -DCMAKE_BUILD_TYPE=Release \
		&& $(MAKE)
	$(QUIET)date > $(toolchain)

.PHONY: distclean-toolchain
distclean-toolchain:
	$(QUIET)$(RM) $(toolchain)
	$(QUIET)$(RM) -r $(LlvmBuild)

DistcleanTargets += distclean-toolchain


#
# Git
#
BuildTargets += $(root)/.git/hooks/pre-push

# Install pre-commit hook:
$(root)/.git/hooks/pre-push: $(root)/tools/pre-push
	@echo '  GIT     $@'
	$(QUIET)cp $< $@


#
# Tidy up
#
.PHONY: clean distclean
clean: $(CleanTargets)
	$(QUIET)$(RM) $(sort $(CleanFiles))

distclean: clean $(DistcleanTargets)
	$(QUIET)$(RM) $(sort $(DistcleanFiles))

#
# Watch
#
WATCH := $(root)/tools/watchr/watchr.js

.PHONY: watch
watch:
	$(QUIET)$(WATCH)


#
# All
#
all: $(BuildTargets)

.PHONY: help
help:
	@echo "Build targets:"
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
		| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
		| sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' \
		| sed 's/^/    /'

foo:
	@echo "$(lab)/stl/include/ustl/$(patsubst %.o,%,$(notdir $(lab)/stl/test/algorithm.o))"
