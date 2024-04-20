#!/bin/bash

for size in 18 19 20 21 22; do nohup ./mgpu.exe $size 6 256 > ${size}_256.out;done

