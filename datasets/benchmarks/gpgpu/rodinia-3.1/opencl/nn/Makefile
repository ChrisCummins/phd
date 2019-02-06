include ../../common/make.config

#C compiler
CC = g++

SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

CC_FLAGS = -g -O3

EXE = nn

release:
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS)
	
clean: 
	rm -f $(EXE) 
