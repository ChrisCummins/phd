# The default goal is...
.DEFAULT_GOAL = all

# Use V=1 argument for verbose builds
QUIET_  = @
QUIET   = $(QUIET_$(V))

# Assume no out-of-tree builds:
root := $(PWD)

#
# Configuration
#

AWK := awk
EGREP := egrep
GREP := grep
MAKEFLAGS := "-j $(SHELL NPR)"
RM := rm -fv
SED := sed
SHELL := /bin/bash


#
# Tidy up
#

.PHONY: help

help:
	@echo "Build targets:"
	@echo
	@echo "  make all"
	@echo "  make clean"
	@echo "  make distclean"
	@echo "  make test"

BuildTargets =
DontLint =


#
# Tidy up
#

.PHONY: clean

CleanFiles = \
	$(NULL)

DistcleanFiles = \
	$(NULL)

clean:
	$(QUIET)$(RM) $(CleanFiles)

distclean: clean
	$(QUIET)$(RM) $(DistcleanFiles)


#
# LaTeX
#

# Targets:
AutotexTargets = \
	$(root)/docs/2015-msc-thesis/thesis.pdf \
	$(root)/docs/2015-progression-review/document.pdf \
	$(root)/docs/wip-adapt/adapt.pdf \
	$(root)/docs/wip-hlpgpu/hlpgpu.pdf \
	$(root)/docs/wip-outline/outline.pdf \
	$(root)/docs/wip-taco/taco.pdf \
	$(NULL)

.PHONY: $(AutotexTargets)

BuildTargets += $(AutotexTargets)

AutotexDirs = $(dir $(AutotexTargets))
AutotexDepFiles = $(addsuffix .autotex.deps, $(AutotexDirs))
AutotexLogFiles = $(addsuffix .autotex.log, $(AutotexDirs))

# Tools:
AUTOTEX := $(root)/tools/autotex.sh

# Rules:
$(AutotexTargets):
	@$(AUTOTEX) make $(patsubst %.pdf,%,$@)

CleanFiles += $(AutotexTargets) $(AutotexDepFiles) $(AutotexLogFiles)

#
# C++
#

RayTracerDir = $(root)/playground/rt

# Targets:
CxxTargets = \
	$(root)/learn/atc++/myvector \
	$(RayTracerDir)/examples/example1 \
	$(RayTracerDir)/examples/example2 \
	$(NULL)

$(RayTracerDir)/examples/example1: $(RayTracerDir)/src/librt.so

$(RayTracerDir)/examples/example2: \
		$(RayTracerDir)/examples/example2.o \
		$(RayTracerDir)/src/librt.so \
		$(NULL)

$(RayTracerDir)/examples/example2.cpp: \
		$(RayTracerDir)/examples/example2.rt \
		$(RayTracerDir)/scripts/mkscene.py
	@echo "  MKSCENE  $@"
	$(QUIET)$(RayTracerDir)/scripts/mkscene.py $< $@ >/dev/null

DontLint += $(RayTracerDir)/examples/example2.cpp

RayTracerSources = \
	$(RayTracerDir)/src/graphics.cpp \
	$(RayTracerDir)/src/lights.cpp \
	$(RayTracerDir)/src/objects.cpp \
	$(RayTracerDir)/src/profiling.cpp \
	$(RayTracerDir)/src/random.cpp \
	$(RayTracerDir)/src/renderer.cpp \
	$(NULL)

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

RayTracerObjects = $(patsubst %.cpp, %.o, $(RayTracerSources))

$(RayTracerDir)/examples_CxxFlags = -I$(RayTracerDir)/include
$(RayTracerDir)/examples_LdFlags = -ltbb
$(RayTracerDir)/src_CxxFlags = -I$(RayTracerDir)/include

$(RayTracerDir)/src/librt.so: $(RayTracerObjects)
	@echo '  LD       $@'
	$(QUIET)$(CXX) $(CxxFlags) $(LdFlags) -fPIC -shared $? -o $@

BuildTargets += $(CxxTargets)

CxxObjects = $(addsuffix .o, $(CxxTargets))
CxxSources = $(addsuffix .cpp, $(CxxTargets))

CxxTargets: $(CxxObjects)
CxxObjects: $(CxxSources)

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

# Tools:
CPPLINT := $(root)/tools/cpplint.py
CXX := $(root)/tools/llvm/build/bin/clang++

# Rules:
%.o: %.cpp
	@echo '  CXX      $@'
	$(QUIET)$(CXX) $(CxxFlags) \
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags) \
		$< -c -o $@
	$(QUIET)if [[ -z "$(filter $<,$(DontLint))" ]]; then \
		$(CPPLINT) $(CxxLintFlags) $< 2>&1 \
			| grep -v '^Done processing\|^Total errors found: ' \
			| tee $<.lint; \
	fi


#
# C
#

# Targets:
CTargets = \
	$(root)/learn/expert_c/cdecl \
	$(root)/learn/expert_c/computer_dating \
	$(NULL)


BuildTargets += $(CTargets)

CObjects = $(addsuffix .o, $(CTargets))
CSources = $(addsuffix .c, $(CTargets))

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
	@echo '  CC       $@'
	$(QUIET)$(CC) $(CFlags) $< -c -o $@


#
# Linker
#

# Linker flags:
LdFlags = \
	$(NULL)

# Rules:
%: %.o
	@echo '  LD       $@'
	$(QUIET)$(CXX) $(CxxFlags) $(LdFlags) \
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags) \
		$($(patsubst %/,%,$@)_LdFlags) \
		$($(patsubst %/,%,$(dir $@))_LdFlags) \
		$^ -o $@

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

$(CxxObjects) $(CObjects): .bootstrapped


#
# All
#

build: $(BuildTargets)

all: build
