compile: add.cpp main.c
	g++ -fpic -shared add.cpp -o libadd.so
	gcc main.c -L. -ladd -o main

run:
	LD_LIBRARY_PATH=$(PWD) ./main
