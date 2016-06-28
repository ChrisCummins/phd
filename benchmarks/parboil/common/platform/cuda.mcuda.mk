# (c) 2011 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Paths
CUDA_PATH=$(MCUDA_PATH)/include

# Programs
CUDACC=$(MCUDA_PATH)/bin/mcc_xmm
CUDALINK=$(LINKER)

# Flags
PLATFORM_CUDACFLAGS=-O3 
PLATFORM_CFLAGS=-O3 -I$(MCUDA_PATH)/include -D__MCUDA__
PLATFORM_CXXFLAGS=-O3 -I$(MCUDA_PATH)/include -D__MCUDA__
PLATFORM_CUDALDFLAGS=-lm -lpthread -L$(MCUDA_PATH)/lib -lmcuda


