RayTracerSources =		\
	$(NULL)
RayTracerHeaders =		\
	math.h			\
	random.h		\
	rt.h			\
	$(NULL)

Sources = main.cc $(RayTracerSources)
Headers = $(RayTracerHeaders)

Binary = rt
Parser = parser.py
Scene = quick.rt.out

CxxFlags = -O2 -Wall -Wextra -std=c++11 -Wno-unused-parameter
LdFlags = -ltbb

# File extension for cpplint tool.
CpplintExtension   = .lint

Objects = $(patsubst %.cc,%.o,$(Sources))

all: quick $(LintFiles)

quick: $(Scene) $(Binary)

clean:
	rm -fv $(Binary) $(Objects) $(Scene)

%.o: %.cc $(Headers) $(Scene)
	g++ $(CxxFlags) -c $<
	@$(call cpplint,$<,$<$(CpplintExtension))

$(Binary): $(Objects)
	g++ $(LdFlags) $^ -o $@

%.rt.out: %.rt $(Parser) scene.rt
	./$(Parser) $< $@

# The cpplint script checks an input source file and enforces the
# style guidelines set out in:
#
#   http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#
CPPLINT = cpplint
LintFiles = $(addsuffix $(CpplintExtension),$(Sources) $(Headers))
MOSTLYCLEANFILES += $(LintFiles)

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
