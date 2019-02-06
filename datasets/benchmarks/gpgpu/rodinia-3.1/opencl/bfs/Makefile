include ../../common/make.config

#C compiler
CC = g++

SRC = bfs.cpp timer.cc

CC_FLAGS = -g -O3

EXE = bfs

release:$(SRC)
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS)

errmsg:$(SRC)
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D ERRMSG PTX_MSG

ptx:$(SRC)
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D PROFILING

res:$(SRC)
	$(CC) $(CC_FLAGS) $(OPENMP) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D RES_MSG

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS)

run:
	./$(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
