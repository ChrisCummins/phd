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
CppLintTargets =
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
	@echo '  LINT     $1.lint'
	$(QUIET)if [[ -z "$(filter $1, $(DontLint))" ]]; then \
		$(CPPLINT) $(CxxLintFlags) $1 2>&1 \
			| grep -v '^Done processing\|^Total errors found: ' \
			| tee $1.lint; \
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
	$(root)/docs/2015-msc-thesis/thesis.pdf \
	$(root)/docs/2015-progression-review/document.pdf \
	$(root)/docs/wip-adapt/adapt.pdf \
	$(root)/docs/wip-hlpgpu/hlpgpu.pdf \
	$(root)/docs/wip-outline/outline.pdf \
	$(NULL)


#
# learn/
#
CxxTargets += \
	$(root)/learn/atc++/myvector \
	$(NULL)

CTargets += \
	$(root)/learn/expert_c/cdecl \
	$(root)/learn/expert_c/computer_dating \
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

CppLintTargets += $(RayTracerHeaders)

RayTracerSources = $(wildcard $(RayTracerDir)/src/*.cpp)
RayTracerObjects = $(patsubst %.cpp, %.o, $(RayTracerSources))

CleanFiles += $(RayTracerObjects) $(RayTracerLib)

# Project specific flags:
RayTracerCxxFlags = -I$(RayTracerDir)/include
$(RayTracerDir)/src_CxxFlags = $(RayTracerCxxFlags)
$(RayTracerDir)/examples_CxxFlags = $(RayTracerCxxFlags)
$(RayTracerDir)/examples_LdFlags = -ltbb

# Link library:
$(RayTracerLib): $(RayTracerObjects)
	$(call o-link,$@,$?,-fPIC -shared)


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
BuildTargets += $(CTargets)

CObjects = $(addsuffix .o, $(CTargets))
CSources = $(addsuffix .c, $(CTargets))

# Compilation requires bootstrapped toolchain:
$(CObjects): $(root)/.bootstrapped

CTargets: $(CObjects)
CObjects: $(CSources)

CleanFiles += $(CTargets) $(CObjects)

# Compiler flags:
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

# Tools:
CC := $(root)/tools/llvm/build/bin/clang

# Rules:
%.o: %.c
	$(call c-compile-o,$@,$<,\
		$($(patsubst %/,%,$@)_CFlags) \
		$($(patsubst %/,%,$(dir $@))_CFlags))


#
# C++
#
CXX := $(root)/tools/llvm/build/bin/clang++

# Deduce
CxxObjects = $(addsuffix .o, $(CxxTargets))
CxxSources = $(addsuffix .cpp, $(CxxTargets))

# Source -> object -> target
BuildTargets += $(CxxTargets)
CxxTargets: $(CxxObjects)
CxxObjects: $(CxxSources)

# Compilation requires bootstrapped toolchain:
$(CxxObjects): $(root)/.bootstrapped

CleanFiles += $(CxxTargets) $(CxxObjects)

# Compiler flags:
CxxFlags = \
	-O2 \
	-std=c++14 \
	-stdlib=libc++ \
	-isystem $(root)/extern/libcxx/include \
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
CppLintFiles = $(addsuffix .lint, $(CppLintTargets))
BuildTargets += $(CppLintFiles)
CleanFiles += $(CppLintFiles)

%.h.lint: %.h
	$(call cpplint,$<)

#
# Linker
#
LD := $(CXX)

LdFlags =

%: %.o
	$(call o-link,$@,$^,\
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

$(AutotexTargets):
	@$(AUTOTEX) make $(patsubst %.pdf,%,$@)
# Autotex does it's own dependency analysis, so always run it:
.PHONY: $(AutotexTargets)

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

$(Python2SetupTestLogs):
	$(call python-setup-test,$(PYTHON2),$(patsubst %/,%,$(dir $@)))

$(Python2SetupInstallLogs):
	$(call python-setup-install,$(PYTHON2),$(patsubst %/,%,$(dir $@)))

$(Python3SetupTestLogs):
	$(call python-setup-test, $(PYTHON3),$(patsubst %/,%,$(dir $@)))

$(Python3SetupInstallLogs):
	$(call python-setup-install,$(PYTHON3),$(patsubst %/,%,$(dir $@)))

.PHONY: \
	$(Python2SetupInstallLogs) \
	$(Python2SetupTestLogs) \
	$(Python3SetupInstallLogs) \
	$(Python3SetupTestLogs) \
	$(NULL)

TestTargets += $(Python2SetupTestLogs) $(Python3SetupTestLogs)
InstallTargets += $(Python2SetupInstallLogs) $(Python3SetupInstalLogs)

# Clean-up:
Python2CleanDirs = $(sort $(Python2SetupTestDirs) $(Python2SetupInstallDirs))
Python3CleanDirs = $(sort $(Python3SetupTestDirs) $(Python3SetupInstallDirs))

python-clean:
	$(QUIET)$(call python-setup-clean,$(PYTHON2),$(Python2CleanDirs))
	$(QUIET)$(call python-setup-clean,$(PYTHON3),$(Python3CleanDirs))

.PHONY: python-clean

CleanTargets += python-clean

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
BOOTSTRAP := $(root)/tools/bootstrap.sh

$(root)/.bootstrapped: $(BOOTSTRAP)
	@echo "Bootstrapping! Go enjoy a coffee, this will take a while."
	$(QUIET)$(BOOTSTRAP)

bootstrap-clean:
	$(QUIET)$(BOOTSTRAP) clean

.PHONY: bootstrap-clean

DistcleanTargets += bootstrap-clean


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
clean: $(CleanTargets)
	$(QUIET)$(RM) $(sort $(CleanFiles))

distclean: clean $(DistcleanTargets)
	$(QUIET)$(RM) $(sort $(DistcleanFiles))

.PHONY: clean distclean


#
# All
#
all: $(BuildTargets)

help:
	@echo "Build targets:"
	@echo
	@echo "  make all"
	@echo "  make test"
	@echo "  make clean"
	@echo "  make distclean"
	@echo "  sudo make install"

.PHONY: help
