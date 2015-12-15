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

# FIXME: BIBTOOL          := bibtool
# FIXME: TEXTLINT         := textlint
AUTOTEX          := $(root)/tools/autotex.sh
AWK              := awk
BIBER            := biber
CHECKCITES       := checkcites
CLEANBIB         := $(root)/tools/cleanbib.py
CPPLINT          := $(root)/tools/cpplint.py
CC               := $(root)/tools/llvm/build/bin/clang
CXX              := $(root)/tools/llvm/build/bin/clang++
DETEX            := detex
EGREP            := egrep
GREP             := grep
MAKEFLAGS        := "-j $(SHELL NPR)"
PARSE_TEXCOUNT   := $(root)/tools/parse_texcount.py
PDF_OPEN         := open
PDFLATEX         := pdflatex
RM               := rm -fv
SED              := sed
SHELL            := /bin/bash
TEXCOUNT         := $(root)/tools/texcount.pl


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

$(AutotexTargets):
	@$(AUTOTEX) make $(patsubst %.pdf,%,$@)

CleanFiles += $(AutotexTargets) $(AutotexDepFiles) $(AutotexLogFiles)

#
# C++
#

CppTargets = \
	$(root)/learn/atc++/myvector \
	$(NULL)

BuildTargets += $(CppTargets)

CppObjects = $(addsuffix .o, $(CppTargets))
CppSources = $(addsuffix .cpp, $(CppTargets))

CppTargets: $(CppObjects)
CppObjects: $(CppSources)

CleanFiles += $(CppTargets) $(CppObjects)


#
# C++ Linter
#

# File name extension.
CppLintExtension = .lint

# Arguments to --filter flag for cpplint.
CppLintFilters = -legal,-build/c++11,-readability/streams,-readability/todo
CppLintFlags = --root=include --filter=$(CppLintFilters)


#
# C++ Build
#

# Compiler flags.
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

%.o: %.cpp
	@echo '  CXX      $@'
	$(QUIET)$(CXX) $(CxxFlags) $< -c -o $@
	$(QUIET)$(CPPLINT) $(CppLintFlags) $< 2>&1 \
	 	| grep -v '^Done processing\|^Total errors found: ' \
		| tee $<.lint


#
# C
#

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

#
# C Build
#

# Compiler flags.
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
	@echo '  CC       $@'
	$(QUIET)$(CC) $(CFlags) $< -c -o $@


#
# Linker
#

# Linker flags.
LdFlags = \
	$(NULL)

%: %.o
	@echo '  LD       $@'
	$(QUIET)$(CXX) $(CxxFlags) $(LdFlags) $^ -o $@


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

$(CppObjects) $(CObjects): .bootstrapped


build: $(BuildTargets)

all: build
