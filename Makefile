SHELL := /bin/bash

BUILD_DIR := ./bin
CUDA_SRC_DIR := ./kernels
AMD_SRC_DIR := ./kernels


C_SRC_DIR := ./csrc
CHPL_MODULES_DIR := ./modules
CUDA_PATH := $(CUDA_HOME)


CUDA_INCLUDE_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib
LIBRARY_DIR := ./libs
C_SOURCES := $(shell find $(C_SRC_DIR) -name '*.c')


ROCM_DIR := $(CHPL_ROCM_PATH)


CHPL_GPU_DEBUB_FLAGS = -s CPUGPUVerbose=false
CHPL_PERF_FLAGS = --fast --no-bounds-checks



QUEENS_DEBUG_FLAGS = -s queens_checkPointer=true -s timeDistributedIters=true -s infoDistributedIters=true
QUEENS_SINGLE_LOC_CPU_FLAGS = -s avoidMirrored=true
QUEENS_MLOCALE_CPU_FLAGS = -s avoidMirrored=true 

QUEENS_MLOCALE_GPU_FLAGS = -s avoidMirrored=true -s queens_mlocale_parameters_parser.GPU=true



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


queens_singlelocale_cuda: cuda dir
	@echo
	@echo " ### Building Chapel Queens single-locale GPU (CUDA-based) ... ### "
	@echo

	chpl $(QUEENS_SINGLE_LOC_CPU_FLAGS) -s GPUCUDA=true -s GPUAMD=false -L$(LIBRARY_DIR) -lqueens -lutil -M $(CHPL_MODULES_DIR) --fast  queens_GPU_CPU_single_node.chpl -o  $(BUILD_DIR)/queens_CUDA_GPU_CPU_single_node.out
	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)

queens_multilocale_cuda: cuda dir
	@echo
	@echo " ### Building Chapel Queens multi-locale GPU (CUDA-based) ... ### "
	@echo

	chpl $(QUEENS_MLOCALE_GPU_FLAGS) -s GPUCUDA=true -s GPUAMD=false -L$(LIBRARY_DIR) -lqueens -lutil -M $(CHPL_MODULES_DIR) --fast $(QUEENS_DEBUG_FLAGS) queens_CPU_GPU_distributed.chpl -o  $(BUILD_DIR)/queens_CPU_GPU_distributed.out
	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)

queens_singlelocale_amd: amd dir
	@echo
	@echo " ### Building the Chapel-AMD single-locale code... ### "
	@echo

	chpl $(QUEENS_SINGLE_LOC_CPU_FLAGS) -s GPUAMD=true -s GPUCUDA=false -I$(ROCM_DIR)/include/ -L$(LIBRARY_DIR) -lamdqueens -L$(ROCM_DIR)/lib/ -lamdhip64  -M $(CHPL_MODULES_DIR) --fast queens_GPU_CPU_single_node.chpl  -o  $(BUILD_DIR)/queens_AMD_GPU_CPU_single_node.out
	@echo
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)


queens_multilocale_amd: amd dir
	@echo
	@echo " ### Building the Chapel-AMD code... ### "
	@echo

	chpl $(QUEENS_MLOCALE_GPU_FLAGS) -s GPUAMD=true -s GPUCUDA=false -I$(ROCM_DIR)/include/ -L$(LIBRARY_DIR) -lamdqueens -L$(ROCM_DIR)/lib/ -lamdhip64  -M $(CHPL_MODULES_DIR) --fast $(QUEENS_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
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
	$(ROCM_DIR)/bin/hipcc --offload-arch=gfx1032 -O3 $(AMD_SRC_DIR)/AMD_queens_kernels.hip --emit-static-lib -fPIC -o $(LIBRARY_DIR)/libamdqueens.a

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

