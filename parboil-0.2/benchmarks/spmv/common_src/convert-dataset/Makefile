convert-dataset: test.c convert_dataset.o mmio.o
	$(CC) -g test.c convert_dataset.o mmio.o -o convert-dataset
    
convert_dataset.o: convert_dataset.c
	$(CC) -g -c convert_dataset.c -o convert_dataset.o

mmio.o: mmio.c mmio.h
	$(CC) -g -c mmio.c -o mmio.o

clean:
	rm -f *.o convert-dataset