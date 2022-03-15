SHELL := /bin/bash
.ONESHELL:

BUILD_DIR := ./bin
CUDA_SRC_DIR := ./kernels
C_SRC_DIR := ./csrc
GPUITE_HOME  := ./chapel-gpu
CHPL_MODULES_DIR := ./modules
CUDA_PATH := $(CUDA_HOME)

CUDA_INCLUDE_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib
LIBRARY_DIR := ./libs
C_SOURCES := $(shell find $(C_SRC_DIR) -name '*.c')

CHPL_DEBUG_FLAGS = -s queens_checkPointer=false -s timeDistributedIters=true -s infoDistributedIters=true -s CPUGPUVerbose=false

chapel: dir cuda gitgpuite
	@echo 
	@echo " ### Building the Chapel code... ### "
	@echo 

	chpl -L$(LIBRARY_DIR) -lqueens -lutil -lGPUAPI -M $(CHPL_MODULES_DIR) -M $(GPUITE_HOME)/src --fast $(CHPL_DEBUG_FLAGS) $(C_SOURCES) $(GPUITE_HOME)/src/GPUAPI.h main.chpl -o  $(BUILD_DIR)/chop.out

	@echo 
	@echo " ### Compilation done ### "
	$(shell sh ./ncomp.sh)


cuda: dir gitgpuite 
	@echo 
	@echo " ### starting CUDA compilation ### "
	@echo 
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libGPUAPI.so $(GPUITE_HOME)/src/GPUAPI.cu           --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libqueens.so $(CUDA_SRC_DIR)/GPU_queens_kernels.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libutil.so $(CUDA_SRC_DIR)/GPU_aux.cu               --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart

gitgpuite: dir

	if [ ! -d "$(GPUITE_HOME)" ] ; then
		@echo 
		@echo " ### cloning from the GPUITE git repo...  ### "
		@echo 
		git clone "git@github.com:ahayashi/chapel-gpu.git" "$(GPUITE_HOME)"
	fi


dir:
	@echo 
	@echo " ### creating directories ### "
	@echo 
	mkdir -p $(LIBRARY_DIR)
	mkdir -p $(BUILD_DIR)


.PHONY: clean
clean:
	$(RM) -Rf $(GPUITE_HOME)
	$(RM) $(LIBRARY_DIR)/*.so
	$(RM) $(BUILD_DIR)/chop.out
	$(RM) $(BUILD_DIR)/chop.out_real