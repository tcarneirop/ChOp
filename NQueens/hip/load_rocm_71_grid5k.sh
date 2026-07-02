#!/bin/bash
unset MODULEPATH
module use /grid5000/guix-modules/x86_64/latest
ml rocm-cmake/7.1.1
ml rocm-hipcc/7.1.1
ml rocm-hip-runtime/7.1.1
ml rocm-smi-lib/7.1.1
ml rocm-smi-lib-bin/7.1.1
ml rocm-toolchain/7.1.1

echo -e "##########################################################"
echo -e "############ ROCM 7.1 modules are loaded  ################" 
echo -e "##########################################################"
