include ../../../common/make.config

INC = -I../common

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

EXE = lud

$(EXE): lud.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) lud.cpp ../common/common.c $(INC) $(LIB) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(EXE)
