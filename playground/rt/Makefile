#
#                       rt top level Makefile
#

# The default goal is...
.DEFAULT_GOAL = all


###############
# Portability #
###############
# Program paths.
export CPPLINT := cpplint
export CXX     := g++
export LD      := g++
export RM      := rm -rf
export SHELL   := /bin/bash


#################
# Build options #
#################
# Compile-time flags.
CxxFlags =			\
	-O2			\
	-pedantic		\
	-Wall			\
	-Wextra			\
	-std=c++11		\
	-Wno-unused-parameter	\
	$(NULL)

# Link-time flags.
LdFlags =			\
	-ltbb			\
	$(NULL)

# Compiler warnings.
CxxWarnings =			\
	cast-align		\
	cast-qual		\
	ctor-dtor-privacy	\
	disabled-optimization	\
	format=2		\
	frame-larger-than=1024	\
	init-self		\
	inline			\
	larger-than=2048	\
	logical-op		\
	missing-declarations	\
	missing-include-dirs	\
	no-div-by-zero		\
	no-main			\
	noexcept		\
	old-style-cast		\
	overloaded-virtual	\
	padded			\
	redundant-decls		\
	shadow			\
	sign-conversion		\
	sign-promo		\
	stack-usage=1024	\
	strict-null-sentinel	\
	strict-overflow=5	\
	switch-default		\
	undef			\
	write-strings		\
	$(NULL)
CxxFlags += $(addprefix -W,$(CxxWarnings))


###########
# Targets #
###########
RayTracerSources =		\
	image.cc		\
	lights.cc		\
	objects.cc		\
	random.cc		\
	rt.cc			\
	$(NULL)
RayTracerHeaders =		\
	camera.h		\
	graphics.h		\
	image.h			\
	lights.h		\
	math.h			\
	random.h		\
	rt.h			\
	scene.h			\
	$(NULL)

Sources = main.cc $(RayTracerSources)
Headers = $(RayTracerHeaders)

Binary = rt
Parser = parser.py
Scene = quick.rt.out

Objects = $(patsubst %.cc,%.o,$(Sources))
CleanFiles = $(Binary) $(Objects) $(Scene)


##########
# Linter #
##########

CpplintExtension = .lint

# The cpplint script checks an input source file and enforces the
# style guidelines set out in:
#
#   http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#
CPPLINT = cpplint
LintFiles = $(addsuffix $(CpplintExtension),$(Sources) $(Headers))
CleanFiles += $(LintFiles)

# Arguments to --filter flag for cpplint.
CpplintFilters = -legal,-build/c++11,-readability/streams,-readability/todo

# Explicit target for creating lint files:
$(LintFiles): %$(CpplintExtension): %
	@$(call cpplint,$<,$@)

# Function for generating lint files.
define cpplint
$(CPPLINT) --filter=$(CpplintFilters) $1 2>&1	 		\
	| grep -v '^Done processing\|^Total errors found: ' 	\
	| tee $2
endef


#########
# Rules #
#########

all: quick $(LintFiles)

quick: $(Scene) $(Binary)

.PHONY: clean

clean:
	$(RM) $(CleanFiles)

%.o: %.cc $(Headers) $(Scene)
	$(CXX) $(CxxFlags) -c $<
	@$(call cpplint,$<,$<$(CpplintExtension))

$(Binary): $(Objects)
	$(LD) $(LdFlags) $^ -o $@

%.rt.out: %.rt $(Parser) scene.rt
	./$(Parser) $< $@
