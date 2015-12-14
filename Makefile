# The default goal is...
.DEFAULT_GOAL = all

# Use V=1 argument for verbose builds
QUIET_  = @
QUIET   = $(QUIET_$(V))


#
# Configuration
#

# FIXME: BIBTOOL          := bibtool
# FIXME: TEXTLINT         := textlint
AUTOTEX          := tools/autotex.sh
AWK              := awk
BIBER            := biber
CHECKCITES       := checkcites
CLEANBIB         := tools/cleanbib.py
CPPLINT          := tools/cpplint.py
CXX              := clang++
DETEX            := detex
EGREP            := egrep
GREP             := grep
MAKEFLAGS        := "-j $(SHELL NPR)"
PARSE_TEXCOUNT   := tools/parse_texcount.py
PDF_OPEN         := open
PDFLATEX         := pdflatex
RM               := rm -fv
SED              := sed
SHELL            := /bin/bash
TEXCOUNT         := texcount


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
	docs/2015-msc-thesis/thesis.pdf \
	docs/2015-progression-review/document.pdf \
	docs/wip-adapt/adapt.pdf \
	docs/wip-hlpgpu/hlpgpu.pdf \
	docs/wip-outline/outline.pdf \
	docs/wip-taco/taco.pdf \
	$(NULL)
.PHONY: $(AutotexTargets)

BuildTargets += $(AutotexTargets)

AutotexDirs = $(dir $(AutotexTargets))
AutotexDepFiles = $(addsuffix .autotex.deps, $(AutotexDirs))
AutotexLogFiles = $(addsuffix .autotex.log, $(AutotexDirs))

$(AutotexTargets):
	$(QUIET)$(AUTOTEX) make $(patsubst %.pdf,%,$@)

CleanFiles += $(AutotexTargets) $(AutotexDepFiles) $(AutotexLogFiles)

#
# C++ Linter
#

# File name extension.
CpplintExtension = .lint

# Arguments to --filter flag for cpplint.
CpplintFilters = -legal,-build/c++11,-readability/streams,-readability/todo
CpplintFlags = --root=include --filter=$(CpplintFilters)


#
# C++ build
#

# Compiler flags.
CxxFlags = \
	-O2 \
	-std=c++14 \
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
	@echo '  CXX      $(notdir $@)'
	$(QUIET)$(CXX) $(CxxFlags)
	$(QUIET)$(CPPLINT) $(CpplintFlags) $< 2>&1 \
		| grep -v '^Done processing\|^Total errors found: ' \
		| tee $@


#
# C++ Linker
#

# Linker flags.
LdFlags = \
	$(NULL)

%: %.o
	@echo '  CXXLD    $(notdir $@)'
	$(QUIET)$(CXX) $(CxxFlags) $(LdFlags) $^ -o $@


#
# Testing
#

test:


build: $(BuildTargets)

all: build
