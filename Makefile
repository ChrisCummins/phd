# The default goal is...
.DEFAULT_GOAL = all

# Use V=1 argument for verbose builds
QUIET_  = @
QUIET   = $(QUIET_$(V))

# Assume no out-of-tree builds:
root := $(PWD)

SHELL := /bin/bash
NPROC := 4

space :=
space +=

#
# Configuration
#
AWK := awk
EGREP := egrep
GREP := grep
MAKEFLAGS := -j$(NPROC)
RM := rm -fv
SED := sed

# Targets:
AutotexTargets =
BuildTargets =
CleanFiles =
CTargets =
CxxTargets =
DistcleanFiles =
DontLint =

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
	@echo '  CC      $1'
	$(QUIET)$(CC) $(CFlags) $3 $2 -c -o $1
endef

# Compile C++ sources to object file
#
# Arguments:
#   $1 (str)   Object file
#   $2 (str[]) C++ sources
#   $3 (str[]) Flags for compiler
define cxx-compile-o
	@echo '  CXX     $1'
	$(QUIET)$(CXX) $(CxxFlags) $3 $2 -c -o $1
	$(QUIET)if [[ -z "$(filter $2, $(DontLint))" ]]; then \
		$(CPPLINT) $(CxxLintFlags) $2 2>&1 \
			| grep -v '^Done processing\|^Total errors found: ' \
			| tee $2.lint; \
	fi
endef

# Link object files to executable
#
# Arguments:
#   $1 (str)   Executable
#   $2 (str[]) Object files
#   $3 (str[]) Flags for linker
define o-link
	@echo '  LD      $1'
	$(QUIET)$(LD) $(CxxFlags) $(LdFlags) $3 $2 -o $1
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
	$(root)/docs/wip-taco/taco.pdf \
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
	$(call o-link, $@, $?, -fPIC -shared)


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
$(CObjects): .bootstrapped

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
	$(call c-compile-o, $@, $<, \
		$($(patsubst %/,%,$@)_CFlags) \
		$($(patsubst %/,%,$(dir $@))_CFlags))


#
# C++
#
CPPLINT := $(root)/tools/cpplint.py
CXX := $(root)/tools/llvm/build/bin/clang++

# Deduce
CxxObjects = $(addsuffix .o, $(CxxTargets))
CxxSources = $(addsuffix .cpp, $(CxxTargets))

# Source -> object -> target
BuildTargets += $(CxxTargets)
CxxTargets: $(CxxObjects)
CxxObjects: $(CxxSources)

# Compilation requires bootstrapped toolchain:
$(CxxObjects): .bootstrapped

CleanFiles += $(CxxTargets) $(CxxObjects)

# Linter:
CxxLintExtension = .lint
CxxLintFilters = -legal,-build/c++11,-readability/streams,-readability/todo
CxxLintFlags = --root=include --filter=$(CxxLintFilters)

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
	$(call cxx-compile-o, $@, $<, \
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags))


#
# Linker
#
LD := $(CXX)

LdFlags =

%: %.o
	$(call o-link, $@, $^, \
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
# Testing
#
test:


#
# Bootstrapping
#
.bootstrapped: tools/bootstrap.sh
	@echo "Bootstrapping! Go enjoy a coffee, this will take a while."
	$(QUIET)./$<


#
# Tidy up
#
clean:
	$(QUIET)$(RM) $(CleanFiles)

distclean: clean
	$(QUIET)$(RM) $(DistcleanFiles)

.PHONY: clean distclean


#
# All
#
all: $(BuildTargets)

help:
	@echo "Build targets:"
	@echo
	@echo "  make all"
	@echo "  make clean"
	@echo "  make distclean"
	@echo "  make test"
.PHONY: help
