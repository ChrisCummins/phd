#
# GNU Makefile - For usage, run 'make help'.
#
# Copyright (C) 2016  Chris Cummins <chrisc.101@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# TODO: Make a list of all the files for which the compilation process
# spat somehting out to stderr. Then implement an 'all-warn' target
# which rebuilds these files.
#
# Run 'make info' to see build dependencies.
#

# The default goal is...
.DEFAULT_GOAL = all


########################################################################
#                       Runtime configuration

#
# Self-documentation for runtime "arguments". For every configurable
# variable, add a doc string using the following format:
#
#   ArgStrings += "<var>=<values>: <description> (default=<default-val>)"
#
# These arg strings are printed by 'make help'. See also $(DocStrings).
#
ArgStrings =

#
# Verbosity controls. There are three levels of verbosity 0-2, set by
# passing the desired V value to Make:
#
#   V=0 print summary messages
#   V=1 same as V=0, but also print executed commands
#   V=3 same as V=2, but with extra verbosity for build system debugging
#
V_default := 0
V ?= $(V_default)
ArgStrings += "V=[0,1,2]: set verbosity level (default=$(V_default))"

#
# Colour controls:
#
#   C=0 disable colour formatting of messages
#   C=1 enable fancy message formatting
#
C_default := 1
C ?= $(C_default)
ArgStrings += "C=[0,1]: enable colour message formatting (default=$(C_default))"

#
# Debug controls:
#
#   D=0 disable debugging support in compiled executables
#   D=1 enable debugging support in compiled executables
#
D_default := 0
D ?= $(D_default)
ArgStrings += "D=[0,1]: enable debugging in generated files (default=$(D_default))"

#
# Optimisation controls:
#
#   O=0 disable optimisations in compiled executables
#   O=1 (default) enable optimisations in compiled executables
#
O_default := 1
O ?= $(O_default)
ArgStrings += "O=[0,1]: enable optimisations in generated files (default=$(O_default))"

#
# Threading controls:
#
threads_default := 4
nproc := $(shell which nproc 2>&1 >/dev/null && nproc || echo $(threads_default))
threads ?= $(shell echo "$(nproc) * 2" | bc -l)
ArgStrings += "threads=[1+]: set number of build threads (default=$(threads_default))"


__verbosity_1_ = @
__verbosity_1_0 = @

__verbosity_2_ = @
__verbosity_2_0 = @
__verbosity_2_1 = @

V1 = $(__verbosity_1_$(V))
V2 = $(__verbosity_2_$(V))


########################################################################
#                         User Configuration
SUDO ?= sudo

# Install prefix:
PREFIX ?= /usr/local

# Non-configurable;
MAKEFLAGS := -j$(threads)
SHELL = /bin/bash
UNAME := $(shell uname)


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
build := $(root)/.build
cache := $(root)/.cache
toolchain := $(build)/toolchain.build

comma := ,
space :=
space +=

AutotexTargets =
BuildTargets =
CleanFiles =
CleanTargets =
CObjects =
CppLintSources =
CTargets =
CxxObjects =
CxxTargets =
DistcleanFiles =
DistcleanTargets =
DontLint =
InstallTargets =
PyLintSources =
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
# These doc strings are printed by 'make help'. See also $(ArgStrings)
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


# Install a file to a location and set mode. Provides a subset of the
# functionality of GNU install, which Mac OS X doesn't provide.
#
# Arguments:
#   $1 (str) Destination
#   $2 (str) Source
#   $3 (str) Mode
define install
	$(call print-task,INSTALL,$1,$(TaskInstall))
	$(V1)$(SUDO) mkdir -p $(dir $1)
	$(V1)$(SUDO) cp $2 $1
	$(V1)$(SUDO) chmod $3 $1
endef


# Download a remote resource.
#
# Arguments:
#   $1 (str) Target path
#   $2 (str) Source URL
#
define wget
	$(call print-task,FETCH,$1,$(TaskInstall))
	$(V1)mkdir -p $(dir $1)
	$(V1)wget -O $1 $2 &>/dev/null
endef

# Unpack an LLVM Tarball.
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball.
#   $3 (str) Tar arguments.
#
define unpack-tar
	$(call print-task,UNPACK,$2)
	$(V1)mkdir -p $1
	$(V1)tar -xf $2 -C $1 --strip-components=1
endef


# Compile C sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C sources
#   $3 (str[]) Flags for compiler
c-compile-o-cmd = $(CC) $(CFlags)
define c-compile-o
	$(call print-task,CC,$1,$(TaskCompile))
	$(V1)$(c-compile-o-cmd) $3 $2 -c -o $1
endef


# Compile C++ sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C++ sources
#   $3 (str[]) Flags for compiler
cxx-compile-o-cmd = $(CXX) $(CxxFlags)
define cxx-compile-o
	$(call print-task,CXX,$1,$(TaskCompile))
	$(V1)$(cxx-compile-o-cmd) $3 $2 -c -o $1
	$(call clang-tidy,$2,$3)
	$(call cpplint,$2)
endef


# Link object files to executable
#
# Arguments:
#   $1 (str)   Executable
#   $2 (str[]) Object files
#   $3 (str[]) Flags for linker
o-link-cmd = $(LD) $(CxxFlags) $(LdFlags)
define o-link
	$(call print-task,LD,$1,$(TaskLink))
	$(V1)$(o-link-cmd) -o $1 $2 $3
endef


cpplint-cmd = $(CPPLINT) $(CxxLintFlags) $1 2>&1 \
	| grep -v '^Done processing\|^Total errors found: ' \
	| tee $1.cpplint

# Run cpplint on input, generating a .cpplint file.
#
# Arguments:
#   $1 (str) C++ source/header
define cpplint
	$(V2)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		test -z "$(V1)" && echo "$(cpplint-cmd)"; \
		$(cpplint-cmd); \
	fi
endef


pylint-cmd = $(PYLINT) $(PyLintFlags) $2 2>&1 \
	| tee $1

# Run pylint on input.
#
# Arguments:
#   $1 (str) Destination output file.
#   $2 (str) Source python file.
#
define pylint
	$(V2)if [[ -z "$(filter $2, $(DontLint))" ]]; then \
		test -z "$(V1)" && echo "$(pylint-cmd)"; \
		$(pylint-cmd); \
	fi
endef


clang-tidy-cmd = $(CLANGTIDY) $1 \
	-checks=-clang-analyzer-security.insecureAPI.rand \
	-- $(CxxFlags) $2

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
	&& grep -E '^Ran [0-9]+ tests in' $2/.$(strip $1).test.log \
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
	cd $2 && $(SUDO) $(strip $1) ./setup.py install --prefix=$(PREFIX) &> \
	$2/.$(strip $1).install.log \
	|| cat $2/.$(strip $1).install.log

# Run python setup.py install
#
# Arguments:
#   $1 (str) Python executable
#   $2 (str) Source directory
define python-setup-install
	$(call print-task,INSTALL,$1: $2,$(TaskInstall))
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
	$(root)/docs/2015-08-msc-thesis/thesis.pdf \
	$(root)/docs/2015-09-progression-review/document.pdf \
	$(root)/docs/2016-01-adapt/adapt.pdf \
	$(root)/docs/2016-01-hlpgpu/hlpgpu.pdf \
	$(root)/docs/2016-06-pldi/abstract.pdf \
	$(root)/docs/2016-07-acaces/abstract.pdf \
	$(root)/docs/2016-07-pact/pact.pdf \
	$(root)/docs/wip-outline/outline.pdf \
	$(NULL)


#
# extern/
#
extern := $(root)/extern


#
# extern/benchmark
#
GoogleBenchmark = $(build)/benchmark/src/libbenchmark.a
GoogleBenchmark_CxxFlags = \
	-isystem $(extern)/benchmark/include -Wno-global-constructors
GoogleBenchmark_LdFlags = -L$(extern)/benchmark/build/src -lbenchmark

# Build flags
GoogleBenchmarkCMakeFlags = \
	-DCMAKE_BUILD_TYPE=Release
$(GoogleBenchmark)-cmd = \
	cd $(build)/benchmark \
	&& $(ToolchainCmake) $(extern)/benchmark -G Ninja >/dev/null \
	&& $(ToolchainEnv) ninja

$(GoogleBenchmark): $(toolchain)
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)rm -rf $(build)/benchmark
	$(V1)mkdir -p $(build)/benchmark
	$(V1)$($(GoogleBenchmark)-cmd)

googlebenchmark: $(GoogleBenchmark)
DocStrings += "googlebenchmark: build Google benchmark library"

.PHONY: distclean-googlebenchmark
distclean-googlebenchmark:
	$(V1)rm -fv -r $(extern)/benchmark/build

DistcleanTargets += distclean-googlebenchmark


#
# extern/boost
#
BoostVersion := 1.46.1
BoostDir := $(build)/boost
BoostBuild = $(BoostDir)/build
Boost = $(BoostBuild)/include

CachedBoostTarball = $(cache)/boost_$(subst .,_,$(BoostVersion)).tar.gz
BoostUrlBase = http://sourceforge.net/projects/boost/files/boost/$(BoostVersion)/

$(CachedBoostTarball):
	$(call wget,$(CachedBoostTarball),$(BoostUrlBase)$(notdir $(CachedBoostTarball)))

# NOTE: Even if boost build fails, we don't care. We only need it to
# copy the headers over for us.
$(Boost)-cmd = \
	cd $(BoostDir) \
	&& ./bootstrap.sh --prefix=$(BoostBuild) >/dev/null \
	&& ./bjam install >/dev/null || true

$(Boost): $(CachedBoostTarball) $(toolchain)
	$(call unpack-tar,$(BoostDir),$<,-zxf)
	$(call print-task,BUILD,boost,$(TaskMisc))
	$(V1)mkdir -p $(BoostBuild)
	$(V1)$($(Boost)-cmd)

Boost_CxxFlags = -isystem $(BoostBuild)/include
Boost_LdFlags = -L$(BoostBuild)/lib

Boost_filesystem_CxxFlags = $(Boost_CxxFlags)
Boost_filesystem_LdFlags = $(Boost_LdFlags) -lboost_filesystem -lboost_system

boost: $(Boost)
DocStrings += "boost: build Boost library"

distclean-boost-cmd = \
	find $(BoostDir) -name '*.a' -o -name '*.o' 2>/dev/null \
		| grep -v config_test.o | xargs rm -fv

.PHONY: distclean-boost
distclean-boost:
	$(V1)if [ -d $(BoostDir) ]; then cd $(BoostDir) && if [ -f bjam ]; then ./bjam clean &>/dev/null; fi fi
	$(V1)$(distclean-boost-cmd)
DistcleanTargets += distclean-boost


#
# extern/clsmith
#
CLSmith = $(extern)/clsmith/build/CLSmith

$(CLSmith)-cmd = \
	cd $(extern)/clsmith/build \
	&& cmake .. >/dev/null && $(MAKE)

$(CLSmith):
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)mkdir -p $(extern)/clsmith/build
	$(V1)$($(CLSmith)-cmd)

.PHONY: distclean-clsmith
distclean-clsmith:
	$(V1)rm -fv -r $(extern)/clsmith/build

DistcleanTargets += distclean-clsmith


#
# extern/googletest
#
GoogleTest = $(build)/googletest/libgtest.a
GoogleTest_CxxFlags = \
	-isystem $(extern)/googletest/googletest/include \
	$(NULL)
GoogleTest_LdFlags = -lpthread -L$(extern)/googletest-build -lgtest

$(GoogleTest)-cmd = \
	cd $(build)/googletest \
	&& $(ToolchainCmake) $(extern)/googletest/googletest -G Ninja >/dev/null \
	&& $(ToolchainEnv) ninja

$(GoogleTest): $(toolchain)
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)rm -rf $(build)/googletest
	$(V1)mkdir -p $(build)/googletest
	$(V1)$($(GoogleTest)-cmd)

googletest: $(GoogleTest)
DocStrings += "googletest: build Google Test library"

.PHONY: distclean-googletest
distclean-googletest:
	$(V1)rm -fv -r $(extern)/googletest-build

DistcleanTargets += distclean-googletest


#
# extern/intel-tbb
#
intelTbbDir = $(build)/intel-tbb
intelTbbBuildDir = $(intelTbbDir)/build/build_release

intelTbb = $(intelTbbBuildDir)/libtbb.so

CachedTbbTarball = $(cache)/tbb44_20160526oss_src_0.tgz
intelTbbUrlBase = https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/

$(CachedTbbTarball):
	$(call wget,$(CachedTbbTarball),$(intelTbbUrlBase)$(notdir $(CachedTbbTarball)))

$(intelTbb): $(CachedTbbTarball) $(toolchain)
	$(call unpack-tar,$(intelTbbDir),$<,zxf)
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)cd $(intelTbbDir) && $(MAKE) clean >/dev/null
	$(V1)cd $(intelTbbDir) && tbb_build_prefix=build $(MAKE) >/dev/null

intelTbb_CxxFlags = -isystem $(intelTbbDir)/include
intelTbb_LdFlags = -L$(intelTbbBuildDir) -ltbb

.PHONY: distclean-intel-tbb
distclean-intel-tbb:
	$(V1)rm -fv -r $(intelTbbDir)
DistcleanTargets += distclean-intel-tbb


#
# extern/libclc
#
LibclcDir = $(extern)/libclc
Libclc = $(LibclcDir)/utils/prepare-builtins.o

Libclc_CxxFlags = -Dcl_clang_storage_class_specifiers \
	-I$(LibclcDir)/generic/include \
	-include $(LibclcDir)/generic/include/clc/clc.h \
	-target nvptx64-nvidia-nvcl -x cl

$(Libclc)-cmd = \
	cd $(LibclcDir) && ./configure.py \
	--with-llvm-config=$(LlvmBuild)/bin/llvm-config && $(MAKE)

$(Libclc): toolchain
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)$($(Libclc)-cmd)

.PHONY: distclean-libclc
distclean-libclc:
	$(V1)cd $(LibclcDir) && if [ -f Makefile ]; then $(MAKE) clean; fi

DistcleanTargets += distclean-libclc


#
# extern/opencl
#
OpenCL_CFlags = -I$(extern)/opencl/include
OpenCL_CxxFlags = $(OpenCL_CFlags)
ifeq ($(UNAME), Linux)
OpenCL_LdFlags = -lOpenCL
else
OpenCL_LdFlags = -framework OpenCL
endif
OpenCL = $(extern)/opencl/include/cl.hpp
$(OpenCL): toolchain


#
# extern/triSYCL
#
# TriSYCL_CxxFlags = $(Boost_CxxFlags) -isystem $(extern)/triSYCL/include
# TriSYCL = $(extern)/triSYCL/include/CL/sycl.hpp


#
# lab/
#
lab := $(root)/lab


#
# lab/lm
#
LmHeaders = $(filter-out $(wildcard $(lab)/lm/include/lm/*.cpplint),$(wildcard $(lab)/lm/include/lm/*))
CppLintSources += $(LmHeaders)
Lm_CxxFlags = -I$(lab)/lm/include

# Lm unit tests:
LmTestsSources = $(wildcard $(lab)/lm/tests/*.cpp)
LmTestsObjects = $(patsubst %.cpp,%.o,$(LmTestsSources))
CxxObjects += $(LmTestsObjects)
$(LmTestsObjects): $(LmHeaders) $(phd) $(GoogleTest)
$(lab)/lm/tests/%.o: $(lab)/lm/tests/%.cpp

$(lab)/lm/tests/tests: $(LmTestsObjects)
CxxTargets += $(lab)/lm/tests/tests
$(lab)/lm/tests_CxxFlags = $(Lm_CxxFlags) $(phd_CxxFlags)
$(lab)/lm/tests_LdFlags = $(phd_LdFlags)

# Lm benchmarks:
LmBenchmarksSources = $(wildcard $(lab)/lm/benchmarks/*.cpp)
LmBenchmarksObjects = $(patsubst %.cpp,%.o,$(LmBenchmarksSources))
CxxObjects += $(LmBenchmarksObjects)
$(LmBenchmarksObjects): $(LmHeaders) $(phd) $(GoogleBenchmark)
$(lab)/lm/benchmarks/%.o: $(lab)/lm/benchmarks/%.cpp

$(lab)/lm/benchmarks/benchmarks: $(LmBenchmarksObjects)
CxxTargets += $(lab)/lm/benchmarks/benchmarks
$(lab)/lm/benchmarks_CxxFlags = $(Lm_CxxFlags) $(phd_CxxFlags)
$(lab)/lm/benchmarks_LdFlags = $(phd_LdFlags)


#
# lab/ml
#
# FIXME: Check link errors for rewriter
# CxxTargets += $(lab)/ml/rewriter

$(lab)/ml/rewriter.o_CxxFlags = $(ClangLlvm_CxxFlags)
$(lab)/ml/rewriter_LdFlags = $(ClangLlvm_LdFlags)


#
# lab/patterns
#
PatternsHeaders = $(wildcard $(lab)/patterns/*.hpp)
PatternsCxxSources = $(wildcard $(lab)/patterns/*.cpp)
PatternsCxxObjects = $(patsubst %.cpp,%.o,$(PatternsCxxSources))
CxxObjects += $(PatternsCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(PatternsCxxSources))

$(lab)/patterns_CxxFlags = $(phd_CxxFlags)
$(lab)/patterns_LdFlags = $(phd_LdFlags)
$(PatternsCxxObjects): $(phd) $(PatternsHeaders)


#
# lab/stl/
#
StlComponents = \
	algorithm \
	array \
	forward_list \
	list \
	map \
	set \
	stack \
	type_traits \
	unordered_map \
	vector \
	$(NULL)

StlHeaders = $(addprefix $(lab)/stl/include/ustl/,$(StlComponents))
CppLintSources += $(StlHeaders)
Stl_CxxFlags = -I$(lab)/stl/include

# Stl unit tests:
StlTestsSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/tests/,$(StlComponents)))
StlTestsObjects = $(patsubst %.cpp,%.o,$(StlTestsSources))
CxxObjects += $(StlTestsObjects)
$(StlTestsObjects): $(StlHeaders) $(phd) $(GoogleTest)
$(lab)/stl/tests/%.o: $(lab)/stl/tests/%.cpp

$(lab)/stl/tests/tests: $(StlTestsObjects)
CxxTargets += $(lab)/stl/tests/tests
$(lab)/stl/tests_CxxFlags = $(Stl_CxxFlags) $(phd_CxxFlags)
$(lab)/stl/tests_LdFlags = $(phd_LdFlags)

# Stl benchmarks:
StlBenchmarksSources = $(addsuffix .cpp,\
	$(addprefix $(lab)/stl/benchmarks/,$(StlComponents)))
StlBenchmarksObjects = $(patsubst %.cpp,%.o,$(StlBenchmarksSources))
CxxObjects += $(StlBenchmarksObjects)
$(StlBenchmarksObjects): $(StlHeaders) $(phd)
$(lab)/stl/benchmarks/%.o: $(lab)/stl/benchmarks/%.cpp

$(lab)/stl/benchmarks/benchmarks: $(StlBenchmarksObjects)
CxxTargets += $(lab)/stl/benchmarks/benchmarks
$(lab)/stl/benchmarks_CxxFlags = $(Stl_CxxFlags) $(phd_CxxFlags)
$(lab)/stl/benchmarks_LdFlags = $(phd_LdFlags)


#
# learn/
#
learn := $(root)/learn


#
# learn/atc++/
#
AtcppCxxSources = $(wildcard $(learn)/atc++/*.cpp)
AtcppCxxObjects = $(patsubst %.cpp,%.o,$(AtcppCxxSources))
CxxObjects += $(AtcppCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(AtcppCxxSources))

$(learn)/atc++_CxxFlags = $(GoogleTest_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(learn)/atc++_LdFlags = $(GoogleTest_LdFlags) $(GoogleBenchmark_LdFlags)
$(wildcard $(learn)/atc++/%.o): $(GoogleTest)


#
# learn/boost/
#
LearnBoostCxxSources = $(wildcard $(learn)/boost/*.cpp)
LearnBoostCxxObjects = $(patsubst %.cpp,%.o,$(LearnBoostCxxSources))
CxxObjects += $(LearnBoostCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(LearnBoostCxxSources))

$(learn)/boost_CxxFlags = $(Boost_filesystem_CxxFlags) -I/opt/local/include
$(learn)/boost_LdFlags = $(Boost_filesystem_LdFlags) -lcrypto -lssl
$(LearnBoostCxxObjects): $(Boost)


#
# learn/challenges/
#

# C++ solutions:
ChallengesCxxSources = $(wildcard $(learn)/challenges/*.cpp)
ChallengesCxxObjects = $(patsubst %.cpp,%.o,$(ChallengesCxxSources))
CxxObjects += $(ChallengesCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(ChallengesCxxSources))

ChallengesCSources = $(wildcard $(learn)/challenges/*.c)
ChallengesCObjects = $(patsubst %.c,%.o,$(ChallengesCSources))
CObjects += $(ChallengesCObjects)
CTargets += $(patsubst %.c,%,$(ChallengesCSources))

DontLint += $(learn)/challenges/009-longest-substr.cpp

$(learn)/challenges/011-big-mandelbrot.o_CxxFlags = $(OpenCL_CxxFlags)
$(learn)/challenges/011-big-mandelbrot_LdFlags = $(OpenCL_LdFlags)
$(learn)/challenges/011-big-mandelbrot.o: $(OpenCL)

$(learn)/challenges_CxxFlags = $(phd_CxxFlags)
$(learn)/challenges_LdFlags = $(phd_LdFlags)
$(ChallengesCObjects) $(ChallengesCxxObjects): \
	$(phd) $(GoogleBenchmark) $(GoogleTest)

#
# learn/ctci/
#

# C++ solutions:
CtCiCxxSources = $(wildcard $(learn)/ctci/*.cpp)
CtCiCxxObjects = $(patsubst %.cpp,%.o,$(CtCiCxxSources))
CxxObjects += $(CtCiCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(CtCiCxxSources))

$(learn)/ctci_CxxFlags = $(GoogleBenchmark_CxxFlags) $(GoogleTest_CxxFlags)
$(learn)/ctci_LdFlags = $(GoogleBenchmark_LdFlags) $(GoogleTest_LdFlags)
$(CtCiCxxObjects): $(phd) $(GoogleBenchmark) $(GoogleTest)


#
# learn/expert_c/
#
ExpertCSources = $(wildcard $(learn)/expert_c/*.c)
CTargets += $(patsubst %.c,%,$(ExpertCSources))

#
# learn/hoocl/
#

# Common files:
HooclCommonHeaders = $(wildcard $(learn)/hoocl/include/*.h)
HooclCCommonSources = $(wildcard $(learn)/hoocl/src/*.c)
HooclCCommonObjects = $(patsubst %.c,%.o,$(HooclCCommonSources))
$(learn)/hoocl/src_CFlags = $(OpenCL_CFlags)
$(HooclCCommonObjects): $(OpenCL) $(HooclCommonHeaders)

# C solutions:
HooclCSources = $(wildcard $(learn)/hoocl/*.c)
HooclCObjects = $(patsubst %.c,%.o,$(HooclCSources))
CObjects += $(HooclCObjects)
CTargets += $(patsubst %.c,%,$(HooclCSources))
$(HooclCObjects): $(OpenCL) $(HooclCCommonObjects) $(HooclCommonHeaders)

# C++ solutions:
HooclCxxSources = $(wildcard $(learn)/hoocl/*.cpp)
HooclCxxObjects = $(patsubst %.cpp,%.o,$(HooclCxxSources))
CxxObjects += $(HooclCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(HooclCxxSources))

$(learn)/hoocl_CFlags = $(OpenCL_CFlags) -I$(learn)/hoocl/include
$(learn)/hoocl_CxxFlags = $(OpenCL_CxxFlags) -I$(learn)/hoocl/include
$(learn)/hoocl_LdFlags = $(OpenCL_LdFlags) $(HooclCCommonObjects)
$(HooclCxxObjects): $(OpenCL)


#
# learn/triSYCL/
#
# LearnTriSYCLCxxSources = $(wildcard $(learn)/triSYCL/*.cpp)
# LearnTriSYCLCxxObjects = $(patsubst %.cpp,%.o,$(LearnTriSYCLCxxSources))
# CxxObjects += $(LearnTriSYCLCxxObjects)
# CxxTargets += $(patsubst %.cpp,%,$(LearnTriSYCLCxxSources))

# $(learn)/triSYCL_CxxFlags = $(TriSYCL_CxxFlags) $(phd_CxxFlags)
# $(learn)/triSYCL_LdFlags = $(TriSYCL_CxxFlags) $(phd_LdFlags)
# $(LearnTriSYCLCxxObjects): $(TriSYCL) $(phd) $(Boost) $(GoogleBenchmark)


#
# learn/pc
#

# C++ solutions:
LearnPcCxxSources = $(wildcard $(learn)/pc/*.cpp)
LearnPcCxxObjects = $(patsubt %.cpp,%.o,$(LearnPcCxxSources))
CxxObjects += $(LearnPcCxxObjects)
CxxTargets += $(patsubst %.cpp,%,$(LearnPcCxxSources))

$(learn)/pc_CxxFlags = $(phd_CxxFlags)
$(learn)/pc_LdFlags = $(phd_LdFlags)
$(LearnPcCxxObjects): $(phd)


#
# playground/
#
playground := $(root)/playground

#
# playground/r/
#
CxxTargets += $(playground)/r/main

#
# playground/rt/
#
RayTracerDir = $(playground)/rt
RayTracerLib = $(RayTracerDir)/src/librt.so

RayTracerBins = \
	$(RayTracerDir)/examples/example1 \
	$(RayTracerDir)/examples/example2 \
	$(NULL)
CxxTargets += $(RayTracerBins)

$(RayTracerDir)/examples/example1: $(RayTracerLib)

$(RayTracerDir)/examples/example2: \
		$(RayTracerDir)/examples/example2.o \
		$(RayTracerLib) \
		$(NULL)

$(RayTracerDir)/examples/example2.cpp: $(RayTracerDir)/examples/example2.rt

# Generate scene files using mkscene script.
%.cpp: %.rt $(RayTracerDir)/scripts/mkscene.py
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
CxxObjects += $(RayTracerObjects)

$(RayTracerObjects) $(addsuffix .o,$(RayTracerBins)): \
	$(RayTracerHeaders) $(intelTbb)

# Project specific flags:
RayTracerCxxFlags = -fPIC -I$(RayTracerDir)/include
$(RayTracerDir)/src_CxxFlags = $(RayTracerCxxFlags)
$(RayTracerDir)/examples_CxxFlags = $(intelTbb_CxxFlags) $(RayTracerCxxFlags)
$(RayTracerDir)/examples_LdFlags = $(intelTbb_LdFlags) -lrt -L$(dir $(RayTracerLib))

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
src = $(root)/src

#
# src/labm8
#
Python2SetupTestDirs += $(src)/labm8
Python2SetupInstallDirs += $(src)/labm8
# Python3SetupTestDirs += $(src)/labm8
Python3SetupInstallDirs += $(src)/labm8
PyLintSources += $(wildcard $(src)/labm8/labm8/*.py)


#
# src/omnitune
#
# Python2SetupTestDirs += $(src)/omnitune
Python2SetupInstallDirs += $(src)/omnitune
PyLintSources += $(wildcard $(src)/omnitune/omnitune/*.py)


#
# src/phd
#
phdSrc = $(src)/phd/src
phdInclude = $(src)/phd/include

phd = $(phdSrc)/libphd.so

phdCxxSources = $(wildcard $(phdSrc)/*.cpp)
phdCxxObjects = $(patsubst %.cpp,%.o,$(phdCxxSources))
CxxObjects += $(phdCxxObjects)
$(phdCxxObjects): $(GoogleBenchmark) $(GoogleTest)

phdCxxHeaders = $(filter-out %.cpplint,$(wildcard $(phdInclude)/*))
CppLintSources += $(phdCxxHeaders)

# Flags to build phd.
$(phdSrc)_CxxFlags = \
	-I$(phdInclude) $(GoogleTest_CxxFlags) $(GoogleBenchmark_CxxFlags)
$(phdSrc)_LdFlags = \
	$(GoogleTest_LdFlags) $(GoogleBenchmark_LdFlags)

# Build phd.
$(phd): $(phdCxxObjects)
	$(call o-link,$@,$(phdCxxObjects),-fPIC -shared)

# Flags to build against phd.
phd_CxxFlags = $($(phdSrc)_CxxFlags)
phd_LdFlags = $($(phdSrc)_LdFlags)


# thesis/
AutotexTargets += $(root)/thesis/thesis.pdf

# tools/
tools = $(root)/tools

pgit = $(PREFIX)/bin/pgit

$(pgit): $(root)/tools/pgit
	$(call install,$@,$<,0755)

InstallTargets += $(pgit)

# make_tools/
pmake = $(PREFIX)/bin/pmake

$(pmake): $(tools)/make_tools/pmake
	$(call install,$@,$<,0755)

InstallTargets += $(pmake)


########################################################################
#                         Build rules


#
# C
#
CC := $(build)/llvm/build/bin/clang

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


#
# C++
#
CXX := $(build)/llvm/build/bin/clang++
CLANGTIDY := $(build)/llvm/build/bin/clang-tidy

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
	-isystem $(build)/llvm/ \
	-std=c++1z \
	-stdlib=libc++ \
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


#
# Linker
#
# TODO: Clang picks the linker for us, and will default to using the
# system linker. I would prefer to use LLVM's lld linker, but in
# initial tests I found that it wasn't up to the task. Perhaps with a
# later release I will give this another punt.
LD := $(CXX)

LdFlags =

%: %.o
	$(call o-link,$@,$(filter %.o,$^),\
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags) \
		$($(patsubst %/,%,$@)_LdFlags) \
		$($(patsubst %/,%,$(dir $@))_LdFlags))

.PHONY: print-ld
print-ld:
	$(V2)echo $(o-link-cmd) $($(PMAKE_INVOC_DIR)_CxxFlags) \
		$($(PMAKE_INVOC_DIR)_LdFlags)
DocStrings += "print-ld: print linker invocation"


#
# LaTeX
#
BuildTargets += $(AutotexTargets)

AutotexDirs = $(dir $(AutotexTargets))
AutotexDepFiles = $(addsuffix .autotex.deps, $(AutotexDirs))
AutotexLogFiles = $(addsuffix .autotex.log, $(AutotexDirs))

# Autotex does it's own dependency analysis, so always run it:
.PHONY: $(AutotexTargets)
$(AutotexTargets):
	$(V2)$(root)/make_tools/autotex.sh make $(patsubst %.pdf,%,$@)

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
	$(call python-setup-test,python2,$(patsubst %/,%,$(dir $@)))

$(Python2SetupInstallLogs):
	$(call python-setup-install,python2,$(patsubst %/,%,$(dir $@)))

$(Python3SetupTestLogs):
	$(call python-setup-test,python3,$(patsubst %/,%,$(dir $@)))

$(Python3SetupInstallLogs):
	$(call python-setup-install,python3,$(patsubst %/,%,$(dir $@)))

TestTargets += $(Python2SetupTestLogs) $(Python3SetupTestLogs)
InstallTargets += $(Python2SetupInstallLogs) $(Python3SetupInstallLogs)

# Clean-up:
Python2CleanDirs = $(sort $(Python2SetupTestDirs) $(Python2SetupInstallDirs))
Python3CleanDirs = $(sort $(Python3SetupTestDirs) $(Python3SetupInstallDirs))

.PHONY: clean-python
clean-python:
	$(V1)$(call python-setup-clean,python2,$(Python2CleanDirs))
	$(V1)$(call python-setup-clean,python3,$(Python3CleanDirs))

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


########################################################################
#                            Toolchain

#
# LLVM Toolchain
#
LlvmVersion := 3.8.1
LlvmSrc := $(build)/llvm
LlvmBuild := $(LlvmSrc)/build
LlvmLibDir := $(LlvmBuild)/lib
LlvmConfig := $(LlvmBuild)/bin/llvm-config
LlvmCMakeFlags := \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=true \
	-DLLVM_TARGETS_TO_BUILD=X86 \
	-G Ninja -Wno-dev \
	$(NULL)

Toolchain_CC := $(LlvmBuild)/bin/clang
Toolchain_CXX := $(LlvmBuild)/bin/clang++
ToolchainCxxFlags := \
	-Wno-unused-command-line-argument \
	-stdlib=libc++ \
	$(NULL)
ToolchainEnv := CC=$(Toolchain_CC) CXX=$(Toolchain_CXX) LD_LIBRARY_PATH=$(LlvmLibDir)
ToolchainCmake := $(ToolchainEnv) cmake	-DCMAKE_CXX_FLAGS="$(ToolchainCxxFlags)"

# Flags to build against LLVM + Clang toolchain
ClangLlvm_CxxFlags = \
	$(shell $(LlvmConfig) --cxxflags) \
	-isystem $(shell $(LlvmConfig) --src-root)/tools/clang/include \
	-isystem $(shell $(LlvmConfig) --obj-root)/tools/clang/include \
	-fno-rtti \
	$(NULL)

ClangLlvm_LdFlags = \
	$(shell $(LlvmConfig) --system-libs) \
	-L$(shell $(LlvmConfig) --libdir) \
	-ldl \
	-lclangTooling \
	-lclangToolingCore \
	-lclangFrontend \
	-lclangDriver \
	-lclangSerialization \
	-lclangCodeGen \
	-lclangParse \
	-lclangSema \
	-lclangStaticAnalyzerFrontend \
	-lclangStaticAnalyzerCheckers \
	-lclangStaticAnalyzerCore \
	-lclangAnalysis \
	-lclangARCMigrate \
	-lclangRewriteFrontend \
	-lclangRewrite \
	-lclangEdit \
	-lclangAST \
	-lclangLex \
	-lclangBasic \
	-lclang \
	-ldl \
	$(shell $(LlvmConfig) --libs) \
	-pthread \
	-lLLVMCppBackendCodeGen -lLLVMTarget -lLLVMMC \
	-lLLVMObject -lLLVMCore -lLLVMCppBackendInfo \
	-ldl -lcurses \
	-lLLVMSupport \
	-lcurses \
	-ldl \
	$(NULL)

# Toolchain dependencies:
$(CC) $(CXX): $(toolchain)
$(CTargets) $(CObjects): $(CC)
$(CxxTargets) $(CxxObjects): $(CXX)

LlvmUrlBase := http://llvm.org/releases/$(LlvmVersion)/
CachedLlvmComponents := \
	llvm \
	cfe \
	clang-tools-extra \
	compiler-rt \
	$(NONE)
LlvmTar := -$(LlvmVersion).src.tar.xz

CachedLlvmTarballs = $(addprefix $(cache)/,$(addsuffix $(LlvmTar),$(CachedLlvmComponents)))

# Fetch LLVM tarballs to local cache.
$(cache)/%$(LlvmTar):
	$(call wget,$@,$(LlvmUrlBase)$(notdir $@))

# Unpack an LLVM Tarball.
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball
#
define unpack-llvm-tar
	$(call unpack-tar,$(LlvmSrc)/$1,$(cache)/$2$(LlvmTar),-xf)
endef

# Unpack LLVM tree from cached tarballs.
$(LlvmSrc): $(CachedLlvmTarballs)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
	$(call unpack-llvm-tar,projects/compiler-rt,compiler-rt)

# Build LLVM.
$(LlvmBuild)/bin/llvm-config: $(LlvmSrc)
	$(call print-task,BUILD,LLVM toolchain,$(TaskMisc))
	$(V1)rm -rf $(LlvmBuild)
	$(V1)mkdir -p $(LlvmBuild)
	$(V1)cd $(LlvmBuild) && cmake .. $(LlvmCMakeFlags) >/dev/null
	$(V1)cd $(LlvmBuild) && ninja

$(toolchain): $(LlvmBuild)/bin/llvm-config
	$(V1)date > $(toolchain)

toolchain: $(toolchain)
DocStrings += "toolchain: build toolchain"

.PHONY: clean-toolchain
clean-toolchain:
	$(V1)rm -fv $(toolchain)
	$(V1)rm -fv -r $(LlvmBuild)
DocStrings += "clean-toolchain: remove toolchain build"

.PHONY: distclean-toolchain
distclean-toolchain: clean-toolchain
	$(V1)rm -fv -r $(LlvmSrc)
DocStrings += "distclean-toolchain: remove *all* toolchain files"


#
# Cache
#
.PHONY: clean-cache
clean-cache:
	$(V1)rm -rf $(cache)
DocStrings += "clean-cache: remove local cache in $(cache)"


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
	$(V1)rm -fv $(sort $(CleanFiles))
DocStrings += "clean: remove generated files"

distclean: clean $(DistcleanTargets)
	$(V1)rm -fv $(sort $(DistcleanFiles))
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
DocStrings += "ls-files: lists files which are built by Makefile"


# List all build targets:
.PHONY: ls-targets
ls-targets:
	$(V2)$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
		| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
		| sort --ignore-case \
		| egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
DocStrings += "ls-targets: show all build targets"

# Repo version information:
git-shorthead-cmd := git rev-parse --short --verify HEAD
git-dirty-cmd := git diff-index --quiet HEAD || echo "*"
version-str = phd-$(shell $(git-shorthead-cmd))$(shell $(git-dirty-cmd))

.PHONY: version
version:
	$(V2)echo 'phd version $(version-str)'
DocStrings += "version: show version information"

# Print information. 'make help' helper.
define print-info
	echo $1 | xargs printf "    %-10s $2\n"
endef

print-program-version-cmd = $(shell which $1 &>/dev/null && \
	{ $1 --version 2>&1 | head -n1; } || { echo not found; })

define print-program-version
	$(call print-info,$1,$(print-program-version-cmd))
endef

# Print doc strings:
.PHONY: help
help:
	$(V2)echo "usage: make [argument...] [target...]"
	$(V2)echo
	$(V2)echo "values for arguments:"
	$(V2)echo
	$(V2)(for var in $(ArgStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-20s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
	$(V2)echo "values for targets (default=all):"
	$(V2)echo
	$(V2)(for var in $(DocStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-20s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
	$(V2)echo "host info:"
	$(V2)echo
	$(V2)$(call print-info,name,$(shell uname -n))
	$(V2)$(call print-info,O/S,$(shell uname -o))
	$(V2)$(call print-info,arch,$(shell uname -m))
	$(V2)$(call print-info,threads,$(threads))
	$(V2)echo
	$(V2)echo "build essentials:"
	$(V2)echo
	$(V2)$(call print-program-version,c++)
	$(V2)$(call print-program-version,cmake)
	$(V2)$(call print-program-version,ninja)
	$(V2)$(call print-program-version,pdflatex)
	$(V2)$(call print-program-version,pep8)
	$(V2)$(call print-program-version,python2)
	$(V2)$(call print-program-version,python3)
	$(V2)$(call print-program-version,svn)
	$(V2)echo
