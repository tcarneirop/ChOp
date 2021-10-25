SHELL := /bin/bash
.ONESHELL:

BUILD_DIR := ./bin
KERNELS_DIR := ./kernels
GPUITE_HOME  := ./chapel-gpu
CHPL_MODULES_DIR := ./modules


LIBRARY_DIR := ./libs
CUDA_PATH := /usr/local/cuda
CULIB := $(CUDA_PATH)/lib64
CUDA_INCLUDE_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib



CHPL_DEBUG_FLAGS = -s timeDistributedIters=true -s infoDistributedIters=true -s debugGPUIterator=true 



chapel: dir gitgpuite cuda  
	@echo 
	@echo " ### Building Chapel code... ### "
	@echo 
	chpl -Llibs -lite -lutil -lGPUAPI -M $(CHPL_MODULES_DIR) -M $(GPUITE_HOME)/src --fast $(CHPL_DEBUG_FLAGS) $(GPUITE_HOME)/src/GPUAPI.h newMain.chpl -o $(BUILD_DIR)/ITEQueens.out
	@echo 
	@echo " ### Compilation done ### "



cuda: dir gitgpuite 
	@echo 
	@echo " ### starting CUDA compilation ### "
	@echo 

	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libGPUAPI.so $(GPUITE_HOME)/src/GPUAPI.cu          --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libite.so    $(KERNELS_DIR)/GPU_queens_kernels.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libutil.so   $(KERNELS_DIR)/GPU_aux.cu             --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart

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
	$(RM) $(BUILD_DIR)/ITEQueens.out
	$(RM) $(BUILD_DIR)/ITEQueens.out_real