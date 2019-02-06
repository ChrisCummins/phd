include ../../common/make.config
#C compiler
CC = g++

#OPENCL_DIR = /home/sc5nf/ati-stream-sdk-v2.2-lnx64/
#INC = $(OPENCL_DIR)/include/ 
#LIB = $(OPENCL_DIR)/lib/x86_64/

FLAGS = -O3
SRC = euler3d.cpp

EXE = euler3d

release:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) $(FLAGS)

errmsg:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D ERRMSG $(FLAGS)

ptx:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D PTX_MSG $(FLAGS)

res:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CFLAGS) $(LDFLAGS) -D RES_MSG $(FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
