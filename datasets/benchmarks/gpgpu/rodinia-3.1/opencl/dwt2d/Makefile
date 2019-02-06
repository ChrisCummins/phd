#all:
#	g++ -o components.o -I/usr/local/cuda-5.5/include -c components.cpp
#	g++ -o prog  -I/usr/local/cuda-5.5/include main.cpp components.o -lOpenCL
#
include ../../common/make.config


ifdef OUTPUT
	override OUTPUT = -DOUTPUT
endif


OUTPUT = -DOUTPUT

C_C = g++

default:	dwt2d

components:
	$(C_C) -o components.o -c components.cpp $(CFLAGS) $(LDFLAGS)

dwt2d:
	$(C_C) -o dwt2d \
		main.cpp \
		$(components) \
		$(OUTPUT) \
		$(OCL_LIB) $(CFLAGS) $(LDFLAGS)

clean:	
	rm -rf *.o dwt2d
	rm *.bmp.dwt.*

