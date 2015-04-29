Sources = main.cc
Headers = rt.h

CxxFlags = -O2 -Wall -Wextra -std=c++11 -Wno-unused-parameter
LdFlags = -ltbb

Objects = $(patsubst %.cc,%.o,$(Sources))
Binary = rt

Scene = quick.rt.out

all: quick

quick: $(Scene) $(Binary)

clean:
	rm -fv $(Binary) $(Objects) $(Scene)

%.o: %.cc $(Headers) $(Scene)
	g++ $(CxxFlags) -c $<

$(Binary): $(Objects)
	g++ $(LdFlags) $^ -o $@

%.rt.out: %.rt
	./parser.py $< $@
