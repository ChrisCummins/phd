# (c) 2010 The Board of Trustees of the University of Illinois.

SRCDIR_OBJS=main.o 
LANGUAGE=cuda
APP_LDFLAGS=-lm -lstdc++

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	APP_CFLAGS += -fp-model precise
else
	APP_CFLAGS += -ffast-math
	APP_CXXFLAGS += -ffast-math
	APP_CUDACFLAGS += -Xptxas -dlcm=cg -arch compute_12 --use_fast_math
endif

# To use the MKL library
ifeq ($(mkl), 1)
	APP_LDFLAGS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	INCLUDES += -I/opt/intel/mkl/9.1.023/include
	APP_CFLAGS += -DMKL
endif

