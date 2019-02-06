include ../../common/make.config

#C compiler
CC = g++


FLAGS = -O3
SRC = streamcluster.cpp darwin_pthread_workaround.cc
EXE = streamcluster

release:$(SRC)
	$(CC)  $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -lpthread $(FLAGS)

errmsg:$(SRC)
	$(CC) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -lpthread -D ERRMSG $(FLAGS)
	
ptx:$(SRC)
	$(CC) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -lpthread -D PTX_MSG $(FLAGS)
	
res:$(SRC)
	$(CC) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -lpthread -D RES_MSG $(FLAGS)
	
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
