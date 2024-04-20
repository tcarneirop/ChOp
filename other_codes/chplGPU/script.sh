#!/bin/bash

for size in 18 19 20 21 22; do nohup ./chplGPU --size=$size --initial_depth=6 >> $size.out;done

