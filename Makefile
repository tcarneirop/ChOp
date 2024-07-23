SHELL := /bin/bash

BUILD_DIR := ./bin
CUDA_SRC_DIR := ./kernels
C_SRC_DIR := ./csrc
CHPL_MODULES_DIR := ./modules
CUDA_PATH := $(CUDA_HOME)


CUDA_INCLUDE_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib
LIBRARY_DIR := ./libs
C_SOURCES := $(shell find $(C_SRC_DIR) -name '*.c')

AMD_DIR := /opt/rocm/

CHPL_GPU_DEBUB_FLAGS = -s CPUGPUVerbose=false
CHPL_DEBUG_FLAGS = -s queens_checkPointer=false -s timeDistributedIters=true -s infoDistributedIters=true

CHPL_SINGLE_LOC_CPU_FLAGS = -s avoidMirrored=true -s GPUMAIN=false -s MULTILOCALE=false 
CHPL_MLOCALE_CPU_FLAGS = -s avoidMirrored=true -s GPUMAIN=false -s MULTILOCALE=true -s queens_mlocale_parameters_parser.GPU=false
CHPL_PERF_FLAGS = --fast --no-bounds-checks --target-cpu native

singlelocalecpu: dir
	@echo
	@echo " ### Building Chapel single-locale GPU... ### "
	@echo
	chpl  $(CHPL_SINGLE_LOC_CPU_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out

	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)

multilocalecpu: dir
	@echo
	@echo " ### Building Chapel single-locale GPU... ### "
	@echo
	chpl $(CHPL_MLOCALE_CPU_FLAGS) $(CHPL_DEBUG_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out

	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)


chapelcuda: cuda dir
	@echo
	@echo " ### Building the Chapel-CUDA code... ### "
	@echo

	chpl -s avoidMirrored=true -s GPUCUDA=true -s GPUAMD=false -L$(LIBRARY_DIR) -lqueens -lutil -M $(CHPL_MODULES_DIR) --fast $(CHPL_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
	
	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)
	

chapelamd: amd dir
	@echo 
	@echo " ### Building the Chapel-AMD code... ### "
	@echo 

	chpl -s avoidMirrored=true -s GPUAMD=true -s GPUCUDA=false -I$(AMD_DIR)/include/ -L$(LIBRARY_DIR) -lamdqueens -L$(AMD_DIR)/lib/ -lamdhip64  -M $(CHPL_MODULES_DIR) --fast $(CHPL_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
	
	@echo 
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)
	
cuda: dir
	@echo 
	@echo " ### starting CUDA compilation ### "
	@echo 
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libqueens.so $(CUDA_SRC_DIR)/CUDA_queens_kernels.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libutil.so $(CUDA_SRC_DIR)/GPU_aux.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart

amd: dir
	@echo 
	@echo " ### starting AMD compilation ### "
	@echo 
	hipcc -O3 -DHIP_FAST_MATH -D__HIP_PLATFORM_AMD__  $(CUDA_SRC_DIR)/AMD_queens_kernels.hip --emit-static-lib -fPIC -o $(LIBRARY_DIR)/libamdqueens.a

dir:
	@echo 
	@echo " ### creating directories ### "
	@echo 
	mkdir -p $(LIBRARY_DIR)
	mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	$(RM) $(LIBRARY_DIR)/*.so
	$(RM) $(BUILD_DIR)/chop.out
	$(RM) $(BUILD_DIR)/chop.out_real

