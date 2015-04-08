Sources = main.cc

CxxFlags = -O0 -DDEBUG
LdFlags =

Objects = $(patsubst %.cc,%.o,$(Sources))

%.o: %.cc
	g++ $(CxxFlags) -c $<

rt: $(Objects)
	g++ $(LdFlags) $^ -o $@
