OpenCL_SDK=$(OPENCL_PATH)
INCLUDE=-I${OpenCL_SDK}/include
LIBPATH=-L${OpenCL_SDK}/lib
LIB=-lOpenCL -lm -lcecl
all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${LIB} ${CFILES} -o ${EXECUTABLE} ${LIB}

clean:
	rm -f *~ *.exe
