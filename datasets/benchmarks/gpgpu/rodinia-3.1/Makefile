include common/make.config

RODINIA_BASE_DIR := $(shell pwd)

CUDA_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/cuda
OMP_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/omp
OPENCL_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/opencl

CUDA_DIRS := backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud nn	nw srad streamcluster particlefilter pathfinder mummergpu
OMP_DIRS  := backprop bfs cfd		   heartwall hotspot kmeans lavaMD leukocyte lud nn nw srad streamcluster particlefilter pathfinder mummergpu
OCL_DIRS  := backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud nn	nw srad streamcluster particlefilter pathfinder

all: OPENCL

OPENCL: \
	$(OPENCL_BIN_DIR)/b_tree \
	$(OPENCL_BIN_DIR)/backprop \
    $(OPENCL_BIN_DIR)/bfs \
    $(OPENCL_BIN_DIR)/cfd \
    $(OPENCL_BIN_DIR)/dwt2d \
    $(OPENCL_BIN_DIR)/gaussian \
    $(OPENCL_BIN_DIR)/heartwall \
    $(OPENCL_BIN_DIR)/hotspot \
    $(OPENCL_BIN_DIR)/hybridsort \
    $(OPENCL_BIN_DIR)/kmeans \
    $(OPENCL_BIN_DIR)/lavaMD \
    $(OPENCL_BIN_DIR)/leukocyte \
    $(OPENCL_BIN_DIR)/lud \
    $(OPENCL_BIN_DIR)/myocyte \
    $(OPENCL_BIN_DIR)/nn \
    $(OPENCL_BIN_DIR)/nw \
    $(OPENCL_BIN_DIR)/particlefilter \
    $(OPENCL_BIN_DIR)/pathfinder \
    $(OPENCL_BIN_DIR)/srad \
    $(OPENCL_BIN_DIR)/streamcluster \
    $(NULL)

$(OPENCL_BIN_DIR)/b_tree:
	cd opencl/b_tree;			make;	cp b_tree	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/b_tree

$(OPENCL_BIN_DIR)/backprop:
	cd opencl/backprop;			make;	cp backprop     $(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/backprop

$(OPENCL_BIN_DIR)/bfs:
	cd opencl/bfs;				make;	cp bfs		$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/bfs

$(OPENCL_BIN_DIR)/cfd:
	cd opencl/cfd;				make;	cp euler3d	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/cfd

$(OPENCL_BIN_DIR)/dwt2d:
	cd opencl/dwt2d;                   	make;   cp dwt2d  $(CUDA_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/dwt2d

$(OPENCL_BIN_DIR)/gaussian:
	cd opencl/gaussian;			make;	cp gaussian	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/gaussian

$(OPENCL_BIN_DIR)/heartwall:
	cd opencl/heartwall;		        make;	cp heartwall	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/heartwall

$(OPENCL_BIN_DIR)/hotspot:
	cd opencl/hotspot;			make;	cp hotspot	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/hotspot

$(OPENCL_BIN_DIR)/hybridsort:
	cd opencl/hybridsort;              	make;   cp hybridsort $(CUDA_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/hybridsort

$(OPENCL_BIN_DIR)/kmeans:
	cd opencl/kmeans;			make;	cp kmeans	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/kmeans

$(OPENCL_BIN_DIR)/lavaMD:
	cd opencl/lavaMD;			make;	cp lavaMD	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/lavaMD

$(OPENCL_BIN_DIR)/leukocyte:
	cd opencl/leukocyte;			make;	cp OpenCL/leukocyte	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/leukocyte

$(OPENCL_BIN_DIR)/lud:
	cd opencl/lud/ocl;			make;	cp lud		$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/lud

$(OPENCL_BIN_DIR)/myocyte:
	cd opencl/myocyte;			make;	cp myocyte	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/myocyte

$(OPENCL_BIN_DIR)/nn:
	cd opencl/nn;				make;	cp nn	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/nn

$(OPENCL_BIN_DIR)/nw:
	cd opencl/nw;				make;	cp nw		$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/nw

$(OPENCL_BIN_DIR)/particlefilter:
	cd opencl/particlefilter;		make;	cp OCL_particlefilter_naive OCL_particlefilter_single OCL_particlefilter_double $(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/particlefilter

$(OPENCL_BIN_DIR)/pathfinder:
	cd opencl/pathfinder;			make;	cp pathfinder	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/pathfinder

$(OPENCL_BIN_DIR)/srad:
	cd opencl/srad;				make;	cp srad		$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/srad

$(OPENCL_BIN_DIR)/streamcluster:
	cd opencl/streamcluster;		make;	cp streamcluster	$(OPENCL_BIN_DIR)
.PHONY: $(OPENCL_BIN_DIR)/streamcluster

# Always build benchmarks.
.PHONY: OPENCL

clean: OCL_clean

OCL_clean:
	cd $(OPENCL_BIN_DIR); rm -f *
	for dir in $(OCL_DIRS) ; do cd opencl/$$dir ; make clean ; cd ../.. ; done
