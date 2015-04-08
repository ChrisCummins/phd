Sources = main.cc

CxxFlags = -O0 -DDEBUG -Wall -Wextra
LdFlags =

Objects = $(patsubst %.cc,%.o,$(Sources))

%.o: %.cc
	g++ $(CxxFlags) -c $<

rt: $(Objects)
	g++ $(LdFlags) $^ -o $@
