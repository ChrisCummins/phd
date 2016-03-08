#
# phd Makefile
#
# This repo uses a single monolithic build system to prepare the
# toolchain, compile all executables, documents, etc. This Makefile
# makes **no** guarantee of portability, although in theory it should
# be.
#

# The default goal is...
.DEFAULT_GOAL = all


########################################################################
#                       Runtime configuration
ArgStrings =

#
# Verbosity controls. There are three levels of verbosity 0-2, set by
# passing the desired V value to Make:
#
#   V=0 (default) print summary messages
#   V=1 same as V=0, but print executed commands
#   V=3 same as V=2, but with extra verbosity for build system debugging
#
ifndef V
V = 0
endif
ArgStrings += "V=[0,1,2]: set verbosity level (default=0)"

#
# Colour controls:
#
#   C=0 disable colour formatting of messages
#   C=1 (default) enable fancy message formatting
#
ifndef C
C = 1
endif
ArgStrings += "C=[0,1]: enable colour message formatting (default=1)"

#
# Debug controls:
#
#   D=0 (default) disable debugging support in compiled executables
#   D=1 enable debugging support in compiled executables
#
ifndef D
D = 0
endif
ArgStrings += "D=[0,1]: enable debugging in generated files (default=0)"

#
# Optimisation controls:
#
#   O=0 disable optimisations in compiled executables
#   O=1 (default) enable optimisations in compiled executables
#
ifndef O
O = 1
endif
ArgStrings += "O=[0,1]: enable optimisations in generated files (default=1)"


__verbosity_1_ = @
__verbosity_1_0 = @

__verbosity_2_ = @
__verbosity_2_0 = @
__verbosity_2_1 = @

V1 = $(__verbosity_1_$(V))
V2 = $(__verbosity_2_$(V))


########################################################################
#                        Static Configuration

#
# The number of threads to use:
#
WorkerThreads := 4

SHELL := /bin/bash

AWK := awk
EGREP := egrep
GREP := grep
PYTHON2 := python2
PYTHON3 := python3
RM := rm -fv
SED := sed


# Set number of worker threads.
MAKEFLAGS := -j$(WorkerThreads)


########################################################################
#                         Output & Messages

#
# Output formatting
#
ifeq ($(C),1)
TTYreset = $(shell tput sgr0)
TTYbold = $(shell tput bold)
TTYstandout = $(shell tput smso)
TTYunderline = $(shell tput smul)

TTYblack = $(shell tput setaf 0)
TTYblue = $(shell tput setaf 4)
TTYcyan = $(shell tput setaf 6)
TTYgreen = $(shell tput setaf 2)
TTYmagenta = $(shell tput setaf 5)
TTYred = $(shell tput setaf 1)
TTYwhite = $(shell tput setaf 7)
TTYyellow = $(shell tput setaf 3)
endif  # $(C)

#
# Task types.
#
TaskCompile = $(TTYred)
TaskAux = $(TTYyellow)
TaskLink = $(TTYgreen)
TaskMisc = $(TTYblue)
TaskInstall = $(TTYcyan)

TaskNameLength := 8

#
# Print message.
#
# Arguments:
#   $1 (str) Message body
#   $2       TTY format string (optional)
define print
	@echo '$2$1$(TTYreset)'
endef

# Print task message.
#
# Arguments:
#   $1 (str) Task name (in uppercase)
#   $2 (str) Target path
#   $3       TTY format string (optional)
#
# Example usage:
#
#   $(call print-task,CC,foo.c,$(TaskCompile))
#   $(call print-task,SCRIPT,script.sh)
define print-task
	@printf '  $(TTYbold)$3%-$(TaskNameLength)s$(TTYreset) %s\n' '$1' '$2'
endef


########################################################################
#                             Variables

# Assume no out-of-tree builds:
root := $(PWD)

comma := ,
space :=
space +=

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

#
# Self-documentation. For every user-visible target, add a doc string
# using the following format:
#
#   DocStrings += "<target>: <description>"
#
# These doc strings are printed by 'make help'.
#
DocStrings =


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
	$(call print-task,CC,$1,$(TaskCompile))
	$(V1)$(CC) $(CFlags) $3 $2 -c -o $1
endef


# Compile C++ sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C++ sources
#   $3 (str[]) Flags for compiler
define cxx-compile-o
	$(call print-task,CXX,$1,$(TaskCompile))
	$(V1)$(CXX) $(CxxFlags) $3 $2 -c -o $1
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
	$(call print-task,LD,$1,$(TaskLink))
	$(V1)$(LD) $(CxxFlags) $(LdFlags) $3 $2 -o $1
endef


cpplint-cmd = $(CPPLINT) $(CxxLintFlags) $1 2>&1 \
	| grep -v '^Done processing\|^Total errors found: ' \
	| tee $1.lint

# Run cpplint on input, generating a .lint file.
#
# Arguments:
#   $1 (str) C++ source/header
define cpplint
	$(V2)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		test -z "$(V1)" && echo "$(cpplint-cmd)"; \
		$(cpplint-cmd); \
	fi
endef


clang-tidy-cmd = $(CLANGTIDY) $1 -- $(CxxFlags) $2

# Run clang-tidy on input.
#
# Arguments:
#  $1 (str) source file
#  $2 (str[]) Compilation flags
define clang-tidy
	$(V2)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		test -z "$(V1)" && echo "$(clang-tidy-cmd)"; \
		$(clang-tidy-cmd); \
	fi
endef


python-setup-test-cmd = \
	cd $2 && $(strip $1) ./setup.py test &> $2/.$(strip $1).test.log \
	&& $(GREP) -E '^Ran [0-9]+ tests in' $2/.$(strip $1).test.log \
	|| sed -n -e '/ \.\.\. /,$${p}' $2/.$(strip $1).test.log | \
	grep -v '... ok'

# Run python setup.py test
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-test
	$(call print-task,TEST,$(strip $1) $(strip $2),$(TaskMisc))
	$(V1)$(python-setup-test-cmd)
endef


python-setup-install-cmd = \
	cd $2 && $(strip $1) ./setup.py install &> $2/.$(strip $1).install.log \
	|| cat $2/.$(strip $1).install.log

# Run python setup.py install
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-install
	$(call print-task,INSTALL,$2,$(TaskInstall))
	$(V1)$(python-setup-install-cmd)
endef


python-setup-clean-cmd = cd $$dir && $1 ./setup.py clean >/dev/null

# Run python setup.py clean
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-clean
	$(V2)for dir in $2; do \
		test -z "$(V1)" && echo "$(python-setup-clean-cmd)"; \
		$(python-setup-clean-cmd); \
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
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)mkdir -pv $(extern)/benchmark/build
	$(V1)cd $(extern)/benchmark/build && cmake .. && $(MAKE)

.PHONY: distclean-googlebenchmark
distclean-googlebenchmark:
	$(V1)$(RM) -r $(extern)/benchmark/build

DistcleanTargets += distclean-googlebenchmark


#
# extern/boost
#
Boost = $(extern)/boost/boost/system
BoostDir = $(extern)/boost
BoostConfigDir = $(tools)

Boost_CxxFlags = -I$(extern)/boost/boost
Boost_LdFlags = -L$(BoostDir)/stage/lib

Boost_filesystem_CxxFlags = $(Boost_CxxFlags)
Boost_filesystem_LdFlags = $(Boost_LdFlags) -lboost_filesystem -lboost_system

$(Boost)-cmd = \
	cd $(BoostDir) && \
	./bootstrap.sh --prefix=$(BoostBuild) cxxflags="-stdlib=libc++" stage \
	&& BOOST_BUILD_PATH=$(BoostConfigDir) \
	./b2 --prefix=$(BoostBuild) threading=multi \
	link=static runtime-link=static \
	cxxflags="-stdlib=libc++" linkflags="-stdlib=libc++"

$(Boost):
	$(call print-task,BUILD,boost,$(TaskMisc))
	$(V1)mkdir -pv $(BoostBuild)
	$(V1)$($(Boost)-cmd)

distclean-boost-cmd = \
	cd $(BoostDir) && ./b2 clean \
	&& find . -name '*.a' -o -name '*.o' -exec rm {} \;

.PHONY: distclean-boost
distclean-boost:
	$(V1)$(distclean-boost-cmd)

DistcleanTargets += distclean-boost


#
# extern/googletest
#
GoogleTest = $(extern)/googletest-build/libgtest.a
GoogleTest_CxxFlags = -I$(extern)/googletest/googletest/include
GoogleTest_LdFlags = -L$(extern)/googletest-build -lgtest

$(GoogleTest)-cmd = \
	cd $(extern)/googletest-build \
	&& cmake ../googletest/googletest && $(MAKE)

$(GoogleTest):
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)mkdir -pv $(extern)/googletest-build
	$(V1)$($(GoogleTest)-cmd)

.PHONY: distclean-googletest
distclean-googletest:
	$(V1)$(RM) -r $(extern)/googletest-build

DistcleanTargets += distclean-googletest


#
# extern/opencl
#
OpenCL_CxxFlags = -I$(extern)/opencl/include
OpenCL_LdFlags = -framework OpenCL
OpenCL = $(extern)/opencl/include/cl.hpp


#
# extern/triSYCL
#
TriSYCL_CxxFlags = -I$(extern)/triSYCL/include


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
	list \
	map \
	stack \
	unordered_map \
	vector \
	$(NULL)

StlHeaders = $(addprefix $(lab)/stl/include/ustl/,$(StlComponents))
Stl_CxxFlags = -I$(lab)/stl/include

# Stl unit tests:
StlTestsSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/tests/,$(StlComponents)))
StlTestsObjects = $(patsubst %.cpp,%.o,$(StlTestsSources))
$(StlTestsObjects): $(StlHeaders) $(GoogleBenchmark)
$(lab)/stl/tests/%.o: $(lab)/stl/tests/%.cpp

$(lab)/stl/tests/tests: $(StlTestsObjects)
CxxTargets += $(lab)/stl/tests/tests
$(lab)/stl/tests_CxxFlags = $(Stl_CxxFlags) $(GoogleTest_CxxFlags)
$(lab)/stl/tests_LdFlags = $(GoogleTest_LdFlags)

# Stl benchmarks:
StlBenchmarksSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/benchmarks/,$(StlComponents)))
StlBenchmarksObjects = $(patsubst %.cpp,%.o,$(StlBenchmarksSources))
$(StlBenchmarksObjects): $(StlHeaders) $(GoogleBenchmark)
$(lab)/stl/benchmarks/%.o: $(lab)/stl/benchmarks/%.cpp

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
	$(learn)/atc++/arrays \
	$(learn)/atc++/benchmark-argument-type \
	$(learn)/atc++/constructors \
	$(learn)/atc++/functional \
	$(learn)/atc++/hash_map \
	$(learn)/atc++/inheritance \
	$(learn)/atc++/lambdas \
	$(learn)/atc++/myvector \
	$(learn)/atc++/size-of \
	$(learn)/atc++/smart-ptr \
	$(learn)/atc++/strings \
	$(learn)/atc++/templates \
	$(learn)/atc++/user-input \
	$(learn)/atc++/value-categories \
	$(NULL)

$(learn)/atc++_CxxFlags = $(GoogleTest_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(learn)/atc++_LdFlags = $(GoogleTest_LdFlags) $(GoogleBenchmark_LdFlags)
$(wildcard $(learn)/atc++/%.o): $(GoogleTest)


#
# learn/boost/
#
LearnBoostCxxSources = $(wildcard $(learn)/boost/*.cpp)
LearnBoostCxxObjects = $(patsubst %.cpp,%.o,$(LearnBoostCxxSources))
CxxTargets += $(patsubst %.cpp,%,$(LearnBoostCxxSources))

$(learn)/boost_CxxFlags = $(Boost_filesystem_CxxFlags)
$(learn)/boost_LdFlags = $(Boost_filesystem_LdFlags) -lcrypto -lssl
$(LearnBoostCxxObjects): $(Boost)


#
# learn/challenges/
#

# C++ solutions:
ChallengesCxxSources = $(wildcard $(learn)/challenges/*.cpp)
ChallengesCxxObjects = $(patsubst %.cpp,%.o,$(ChallengesCxxSources))
CxxTargets += $(patsubst %.cpp,%,$(ChallengesCxxSources))

$(learn)/challenges_CxxFlags = \
	$(GoogleBenchmark_CxxFlags) $(GoogleTest_CxxFlags)
$(learn)/challenges_LdFlags = \
	$(GoogleBenchmark_LdFlags) $(GoogleTest_LdFlags)
$(ChallengesCxxObjects): $(GoogleBenchmark) $(GoogleTest)


#
# learn/ctci/
#

# C++ solutions:
CtCiCxxSources = $(wildcard $(learn)/ctci/*.cpp)
CtCiCxxObjects = $(patsubst %.cpp,%.o,$(CtCiCxxSources))
CxxTargets += $(patsubst %.cpp,%,$(CtCiCxxSources))

$(learn)/ctci_CxxFlags = $(GoogleBenchmark_CxxFlags) $(GoogleTest_CxxFlags)
$(learn)/ctci_LdFlags = $(GoogleBenchmark_LdFlags) $(GoogleTest_LdFlags)
$(CtCiCxxObjects): $(GoogleBenchmark) $(GoogleTest)


#
# learn/expert_c/
#
ExpertCSources = $(wildcard $(learn)/expert_c/*.c)
CTargets += $(patsubst %.c,%,$(ExpertCSources))


#
# learn/triSYCL/
#
LearnTriSYCLCxxSources = $(wildcard $(learn)/triSYCL/*.cpp)
LearnTriSYCLCxxObjects = $(patsubst %.cpp,%.o,$(LearnTriSYCLCxxSources))
CxxTargets += $(patsubst %.cpp,%,$(LearnTriSYCLCxxSources))

$(learn)/triSYCL_CxxFlags = \
	$(TriSYCL_CxxFlags) $(GoogleTest_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(learn)/triSYCL_LdFlags = \
	$(TriSYCL_CxxFlags) $(GoogleTest_LdFlags) $(GoogleBenchmark_LdFlags)
$(LearnTriSYCLCxxObjects): $(GoogleBenchmark) $(GoogleTest)


#
# learn/pc
#

# C++ solutions:
LearnPcCxxSources = $(wildcard $(learn)/pc/*.cpp)
LearnPcCxxObjects = $(patsubt %.cpp,%.o,$(LearnPcCxxSources))
CxxTargets += $(patsubst %.cpp,%,$(LearnPcCxxSources))

$(learn)/pc_CxxFlags = $(GoogleTest_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(learn)/pc_LdFlags = $(GoogleTest_LdFlags) $(GoogleBenchmark_LdFlags)
$(LearnPcCxxObjects): $(GoogleBenchmark) $(GoogleTest)


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
	$(call print-task,MKSCENE,$@,$(TaskAux))
	$(V1)$(RayTracerDir)/scripts/mkscene.py $< $@ >/dev/null

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
RayTracerObjects = $(patsubst %.cpp,%.o,$(RayTracerSources))

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

#
# src/labm8
#
Python2SetupTestDirs += $(root)/src/labm8
Python2SetupInstallDirs += $(root)/src/labm8
Python3SetupTestDirs += $(root)/src/labm8
Python3SetupInstallDirs += $(root)/src/labm8


#
# src/omnitune
#
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
	-Wall \
	-Wextra \
	-Wcast-align \
	-Wcast-qual \
	-Wctor-dtor-privacy \
	-Wdisabled-optimization \
	-Wformat=2 \
	-Winit-self \
	-Winline \
	-Wmissing-declarations \
	-Wmissing-include-dirs \
	-Wno-div-by-zero \
	-Wno-main \
	-Wno-missing-braces \
	-Wno-unused-parameter \
	-Wold-style-cast \
	-Woverloaded-virtual \
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

c: $(CTargets)
DocStrings += "c: build all C targets"


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

# Inherit optimisation/debug flags from C config:
CxxOptimisationFlags_$(O) = $(COptimisationFlags_$(O))
CxxOptimisationFlags = $(CxxOptimisationFlags_$(O))

CxxDebugFlags_$(D) = $(CDebugFlags_$(D))
CxxDebugFlags = $(CxxDebugFlags_$(D))

CxxFlags = \
	$(CxxOptimisationFlags) \
	$(CxxDebugFlags) \
	-std=c++1z \
	-stdlib=libc++ \
	-pedantic \
	-Wall \
	-Wextra \
	-Wcast-align \
	-Wcast-qual \
	-Wctor-dtor-privacy \
	-Wdisabled-optimization \
	-Wformat=2 \
	-Winit-self \
	-Winline \
	-Wmissing-declarations \
	-Wmissing-include-dirs \
	-Wno-div-by-zero \
	-Wno-main \
	-Wno-missing-braces \
	-Wno-unused-parameter \
	-Wold-style-cast \
	-Woverloaded-virtual \
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

cpp: $(CxxTargets)
DocStrings += "cpp: build all C++ targets"


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
	$(call print-task,LINT,$@,$(TaskAux))
	$(call cpplint,$<)

lint: $(CppLintTargets)
DocStrings += "lint: build all lint files"


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

# Autotex does it's own dependency analysis, so always run it:
.PHONY: $(AutotexTargets)
$(AutotexTargets):
	$(V2)$(AUTOTEX) make $(patsubst %.pdf,%,$@)

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

tex: $(AutotexTargets)
DocStrings += "tex: build all LaTeX targets"


#
# Python (2 and 3)
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
	$(V1)$(call python-setup-clean,$(PYTHON2),$(Python2CleanDirs))
	$(V1)$(call python-setup-clean,$(PYTHON3),$(Python3CleanDirs))

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
DocStrings += "test: run all tests"


#
# Install
#
install: $(InstallTargets)
DocStrings += "install: install files"


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

$(toolchain)-cmd = \
	cd $(LlvmBuild) \
	&& cmake $(LlvmSrc) -DCMAKE_BUILD_TYPE=Release \
	&& $(MAKE)

$(toolchain):
	$(call print,Bootstrapping! Go enjoy a coffee, this will take a while.)
	$(V1)mkdir -vp $(LlvmBuild)
	$(V1)$($(toolchain)-cmd)
	$(V1)date > $(toolchain)

.PHONY: distclean-toolchain
distclean-toolchain:
	$(V1)$(RM) $(toolchain)
	$(V1)$(RM) -r $(LlvmBuild)

DistcleanTargets += distclean-toolchain

toolchain: $(toolchain)
DocStrings += "toolchain: build toolchain"


#
# Git
#
GitTargets = $(root)/.git/hooks/pre-push
BuildTargets += $(GitTargets)

# Install pre-commit hook:
$(root)/.git/hooks/pre-push: $(root)/tools/pre-push
	$(call print-task,GIT,$@,$(TaskMisc))
	$(V1)cp $< $@

git: $(GitTargets)
DocStrings += "git: configure version control"


#
# Tidy up
#
.PHONY: clean distclean
clean: $(CleanTargets)
	$(V1)$(RM) $(sort $(CleanFiles))
DocStrings += "clean: remove generated files"

distclean: clean $(DistcleanTargets)
	$(V1)$(RM) $(sort $(DistcleanFiles))
DocStrings += "distclean: remove *all* generated files and toolchain"


#
# All.
#
all: $(BuildTargets)
DocStrings += "all: build everything"


#
# Help & documentation
#

# List all build files:
.PHONY: ls-files
ls-files:
	$(V2)$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
	| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
	| sort --ignore-case \
	| grep '^/'
DocStrings += "ls-files: show all files which are built by Makefile"


# List all build targets:
.PHONY: ls-targets
ls-targets:
	$(V2)$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
		| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
		| sort --ignore-case \
		| egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
DocStrings += "ls-targets: show all build targets"


# Print doc strings:
.PHONY: help
help:
	$(V2)echo "make targets:"
	$(V2)echo
	$(V2)(for var in $(DocStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-12s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
	$(V2)echo "make arguments:"
	$(V2)echo
	$(V2)(for var in $(ArgStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-12s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
