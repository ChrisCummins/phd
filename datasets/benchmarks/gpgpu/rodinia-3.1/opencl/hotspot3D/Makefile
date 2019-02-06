CC ?= gcc

OUTPUT = *.out

hotspot3D: 3D.c CL_helper.c CL_helper.h
	$(CC) 3D.c CL_helper.c -O3 -lm $(LIBS) $(CFLAGS) $(LDFLAGS) -o hotspot3D

clean:
	rm -f hotspot3D $(OUTPUT)
