Sources = main.cc
Headers = rt.h

CxxFlags = -O2 -Wall -Wextra -std=c++11 -Wno-unused-parameter
LdFlags = -ltbb

Objects = $(patsubst %.cc,%.o,$(Sources))
Binary = rt

all: $(Binary)

clean:
	rm -fv $(Binary) $(Objects)

%.o: %.cc $(Headers)
	g++ $(CxxFlags) -c $<

$(Binary): $(Objects)
	g++ $(LdFlags) $^ -o $@
