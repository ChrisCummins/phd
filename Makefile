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
CC               := $(PWD)/tools/llvm/build/bin/clang
CXX              := $(PWD)/tools/llvm/build/bin/clang++
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
	@$(AUTOTEX) make $(patsubst %.pdf,%,$@)

CleanFiles += $(AutotexTargets) $(AutotexDepFiles) $(AutotexLogFiles)

#
# C++
#

CppTargets = \
	learn/atc++/myvector \
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
	-isystem $(PWD)/extern/libcxx/include \
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
	@echo '  CXX      $(PWD)/$@'
	$(QUIET)$(CXX) $(CxxFlags) $< -c -o $@
	$(QUIET)$(CPPLINT) $(CppLintFlags) $< 2>&1 \
	 	| grep -v '^Done processing\|^Total errors found: ' \
		| tee $<.lint


#
# C
#

CTargets = \
	learn/expert_c/cdecl \
	learn/expert_c/computer_dating \
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
	@echo '  CC       $(PWD)/$@'
	$(QUIET)$(CC) $(CFlags) $< -c -o $@


#
# Linker
#

# Linker flags.
LdFlags = \
	$(NULL)

%: %.o
	@echo '  LD       $(PWD)/$@'
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
