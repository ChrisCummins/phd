include ../../common/make.config

#C compiler
CC = g++

CC_FLAGS = -g -O3 -Wall

EXE = nw

$(EXE): nw.c
	$(CC) ${KERNEL_DIM} $(CC_FLAGS) -o $(EXE) nw.c $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(EXE)
