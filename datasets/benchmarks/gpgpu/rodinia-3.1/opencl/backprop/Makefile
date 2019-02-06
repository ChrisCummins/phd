include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2

backprop: backprop.o imagenet.o facetrain.o backprop_ocl.o
	$(CC) $(CC_FLAGS) backprop_ocl.o backprop.o imagenet.o facetrain.o -o backprop $(CFLAGS) $(LDFLAGS)

backprop_ocl.o: backprop_ocl.cpp
	$(CC) $(CC_FLAGS) $< -c $(CFLAGS) $(LDFLAGS)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

backprop.o: backprop.c
	$(CC) $(CC_FLAGS) backprop.c -c

facetrain.o: facetrain.c
	$(CC) $(CC_FLAGS) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop *.linkinfo
