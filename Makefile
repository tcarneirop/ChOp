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

QUEENS_DEBUG_FLAGS = -s queens_checkPointer=true -s timeDistributedIters=true -s infoDistributedIters=true
QUEENS_SINGLE_LOC_CPU_FLAGS = -s avoidMirrored=true
QUEENS_MLOCALE_CPU_FLAGS = -s avoidMirrored=true 

CHPL_MLOCALE_GPU_FLAGS = -s avoidMirrored=true -s GPUMAIN=true -s MULTILOCALE=true -s queens_mlocale_parameters_parser.GPU=true

CHPL_PERF_FLAGS = --fast --no-bounds-checks

FSP_COMP_FLAGS = -Llibs csrc/fsp_gen.c csrc/simple_bound.c csrc/johnson_bound.c csrc/aux.c
FSP_DIST_FLAGS = -s timeDistributedIters=true -s infoDistributedIters=true

queens_singlelocale_cpu: dir
	@echo
	@echo " ### Building Chapel Queens single-locale CPU... ### "
	@echo
	chpl  $(QUEENS_SINGLE_LOC_CPU_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) queens_CPU_single_node.chpl -o  $(BUILD_DIR)/queens_mcore.out


	@echo
	@echo " ### Queens-Single Locale -- Compilation done ### "
	$(shell sh ./ncomp.sh)


fsp_singlelocale_cpu: dir
	@echo
	@echo " ### Building Chapel FSP single-locale CPU... ### "
	@echo
	chpl  $(FSP_COMP_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) FSP_CPU_single_node.chpl -o  $(BUILD_DIR)/fsp_mcore.out

	@echo
	@echo " ### FSP-Single Locale -- Compilation done ### "
	$(shell sh ./ncomp.sh)

fsp_multilocale_cpu: dir
	@echo
	@echo " ### Building Chapel FSP multilocale-locale CPU... ### "
	@echo
	chpl  $(FSP_COMP_FLAGS) $(FSP_DIST_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) FSP_CPU_distributed.chpl -o  $(BUILD_DIR)/fsp_mcore.out

	@echo
	@echo " ### FSP-Single Locale -- Compilation done ### "
	$(shell sh ./ncomp.sh)

queens_multilocale_cpu: dir
	@echo
	@echo " ### Building Chapel Queens multi-locale CPU... ### "
	@echo
	chpl $(QUEENS_MLOCALE_CPU_FLAGS) $(QUEENS_DEBUG_FLAGS) -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) queens_CPU_distributed.chpl -o  $(BUILD_DIR)/queens_distributed.out

	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)


chapelcuda: cuda dir
	@echo
	@echo " ### Building the Chapel-CUDA code... ### "
	@echo

	chpl $(CHPL_MLOCALE_GPU_FLAGS) -s GPUCUDA=true -s GPUAMD=false -L$(LIBRARY_DIR) -lqueens -lutil -M $(CHPL_MODULES_DIR) --fast $(QUEENS_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)

chapelamd: amd dir
	@echo
	@echo " ### Building the Chapel-AMD code... ### "
	@echo

	chpl $(CHPL_MLOCALE_GPU_FLAGS) -s GPUAMD=true -s GPUCUDA=false -I$(AMD_DIR)/include/ -L$(LIBRARY_DIR) -lamdqueens -L$(AMD_DIR)/lib/ -lamdhip64  -M $(CHPL_MODULES_DIR) --fast $(QUEENS_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
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

