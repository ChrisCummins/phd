all:
	gcc -O3 ${CFILES} ${CFLAGS} -o ${EXECUTABLE} -lm -lcecl ${LDFLAGS}

clean:
	rm -f *~ *.exe
