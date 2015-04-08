Sources = main.cc

CxxFlags = -O0 -DDEBUG -Wall -Wextra -std=c++11
LdFlags =

Objects = $(patsubst %.cc,%.o,$(Sources))
Binary = rt

all: $(Binary)

clean:
	rm -fv $(Binary) $(Objects)

%.o: %.cc
	g++ $(CxxFlags) -c $<

$(Binary): $(Objects)
	g++ $(LdFlags) $^ -o $@
