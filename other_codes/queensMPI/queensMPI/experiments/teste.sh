#!/bin/bash

for l in 32 16 8 4 2 1
do
	for s in 15 16 17 18 19 20  
	do 
		time (mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np $l -machinefile $OAR_NODEFILE queens $s 4 1) 2>> times/dynamic/$l/$s.txt 
	done
done



for l in 32 16 8 4 2 1
do
	for s in 15 16 17 18 19 20
	do 
		time (mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np $l -machinefile $OAR_NODEFILE queens $s 4 0) 2>> times/static/$l/$s.txt 
	done
done

