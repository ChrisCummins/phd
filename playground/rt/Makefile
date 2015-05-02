RayTracerSources =		\
	$(NULL)
RayTracerHeaders =		\
	math.h			\
	random.h		\
	rt.h			\
	$(NULL)

Sources = main.cc $(RayTracerSources)
Headers = $(RayTracerHeaders)

CxxFlags = -O2 -Wall -Wextra -std=c++11 -Wno-unused-parameter
LdFlags = -ltbb

Objects = $(patsubst %.cc,%.o,$(Sources))
Binary = rt
Parser = parser.py

Scene = quick.rt.out

all: quick

quick: $(Scene) $(Binary)

clean:
	rm -fv $(Binary) $(Objects) $(Scene)

%.o: %.cc $(Headers) $(Scene)
	g++ $(CxxFlags) -c $<

$(Binary): $(Objects)
	g++ $(LdFlags) $^ -o $@

%.rt.out: %.rt $(Parser) scene.rt
	./$(Parser) $< $@
