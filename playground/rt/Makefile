Sources = main.cc

CxxFlags = -O0 -DDEBUG -Wall -Wextra -std=c++11
LdFlags =

Objects = $(patsubst %.cc,%.o,$(Sources))

%.o: %.cc
	g++ $(CxxFlags) -c $<

rt: $(Objects)
	g++ $(LdFlags) $^ -o $@
