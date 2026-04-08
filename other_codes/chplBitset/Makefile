SHELL := /bin/bash

BUILD_DIR := ./bin
CHPL_MODULES_DIR := ./modules

CHPL_DEBUG_FLAGS = -s timeDistributedIters=true -s infoDistributedIters=true

CHPL_PERF_FLAGS = --fast --no-bounds-checks --target-cpu native

multilocalecpu: dir
	@echo
	@echo " ### Building Chapel single-locale GPU... ### "
	@echo
	chpl  -M $(CHPL_MODULES_DIR) $(CHPL_PERF_FLAGS) main.chpl -o  $(BUILD_DIR)/main.out
	@echo
	@echo " ### Compilation done ### "


dir:
	@echo 
	@echo " ### creating directories ### "
	@echo 
	mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	$(RM) $(BUILD_DIR)/main.out
	$(RM) $(BUILD_DIR)/main.out_real

