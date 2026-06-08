
#!/bin/bash

#sudo su

#apt-get install clang lld # clang-12 lld-12 can be added to automatically install the most recent version of the package.


sudo apt-get install clang-12 lldb-12 lld-12

#To install all key packages:

# LLVM

sudo apt-get install libllvm-12-ocaml-dev libllvm12 llvm-12 llvm-12-dev llvm-12-doc llvm-12-examples llvm-12-runtime
# Clang and co

sudo apt-get install clang-12 clang-tools-12 clang-12-doc libclang-common-12-dev libclang-12-dev libclang1-12 clang-format-12 python3-clang-12 clangd-12 clang-tidy-12
# libfuzzer

sudo apt-get install libfuzzer-12-dev
# lldb
	
sudo apt-get install lldb-12
# lld (linker)

sudo apt-get install lld-12
# libc++

sudo apt-get install libc++-12-dev libc++abi-12-dev
# OpenMP

sudo apt-get install libomp-12-dev
# libclc

sudo apt-get install libclc-12-dev
# libunwind

sudo apt-get install libunwind-12-dev
# mlir

sudo apt-get install python-clang

sudo apt-get install  libclang-cpp12-dev 
