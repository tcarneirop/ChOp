#!/bin/bash

testList(){

	locs=(4 2 1)

	for i in ${locs[@]}; do
		
		export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -$i | tr '\n' ' ')

		echo $GASNET_SSH_SERVERS
		./hello -nl $((i+1))

	done

}


tests0806(){

	locs=(10 8 4 2)
	queens=(17 18 19 20 21)
	#queens=(15)
		

	for loc in ${locs[@]}; do
		
		export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -$loc | tr '\n' ' ')
		
		echo $GASNET_SSH_SERVERS

		for q in ${queens[@]}; do

			echo $(date)
			echo "Queens - $q locales: $loc"
			./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))  >> results/$q$loc.txt
			#./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))
		done 

	done
}

tests0906(){


	export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -4 | tr '\n' ' ')
	echo $GASNET_SSH_SERVERS

	echo $(date)
	echo "Queens - 20 locales: 4"
	./bin/fsp.out --size=20 --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl 5  >> results/204.txt



	export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -2 | tr '\n' ' ')
	echo $GASNET_SSH_SERVERS

	echo $(date)
	echo "Queens - 21 locales: 2"
	./bin/fsp.out --size=21 --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl 3  >> results/212.txt
	
			
}


testsl0806(){
	slqueens=(17 18 19)
	#slqueens=(15)
	for q in ${slqueens[@]}; do
		echo "Queens - $q locales: 1"
		./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl1  >> results/sl$q.txt
		#./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl1

	done
}


tests1006(){


	#export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -4 | tr '\n' ' ')
	#echo $GASNET_SSH_SERVERS

	#echo $(date)
	#echo "Queens - 21 locales: 1"
	#./bin/fsp.out --size=20 --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl 5  >> results/204.txt



	export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -2 | tr '\n' ' ')
	echo $GASNET_SSH_SERVERS

	echo $(date)
	echo "Queens - 21 locales: 2"
	./bin/fsp.out --size=21 --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=false --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl 3  >> results/212.txt
	
			
}



testsGPU0806(){
	slqueens=(17 18 19 20)
	#slqueens=(15)
	for q in ${slqueens[@]}; do
		echo "GPUQueens - $q locales: 1"
		./bin/fsp.out --size=$q --lower_bound="queens" --mode="mgpu"  --initial_depth=7 -nl1  >> results/gpu$q.out
			#./bin/fsp.out --size=$q --lower_bound="queens" --mode="mgpu"  --initial_depth=7 -nl1

	done
}



tests1206(){

	locs=(12)
	queens=(17 18 19 20 21)
	#queens=(15)
		

	for loc in ${locs[@]}; do
		
		export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -$loc | tr '\n' ' ')
		
		echo $GASNET_SSH_SERVERS

		for q in ${queens[@]}; do

			echo $(date)
			echo "Queens - $q locales: $loc"
			./bin/fsp.out --size=${q} --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))  >> results/${q}${loc}.txt
			#./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))
		done 

	done
}


testsStatic(){

	locs=(12)
	queens=(21)
		

	for loc in ${locs[@]}; do
		
		#export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -$loc | tr '\n' ' ')
		
		echo $GASNET_SSH_SERVERS

		for q in ${queens[@]}; do

			echo $(date)
			echo "Queens - $q locales: $loc"
			./bin/fsp.out --size=${q} --lower_bound="queens" --mode="improved" --scheduler="static" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl12  >> results/static${q}${loc}.txt
			echo $(date)

		done 

	done
}


testsGuided(){

	locs=(12)
	queens=(17 18 19 20 21)
	#queens=(15)
		

	for loc in ${locs[@]}; do
		
		export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | head -$loc | tr '\n' ' ')
		
		echo $GASNET_SSH_SERVERS

		for q in ${queens[@]}; do

			echo $(date)
			echo "Queens - $q locales: $loc"
			./bin/fsp.out --size=${q} --lower_bound="queens" --mode="improved" --scheduler="guided" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))  >> results/guided${q}${loc}.txt
			#./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))
		done 

	done
}



testsMcore(){

	queens=(17 18 19 20)	

	for q in ${queens[@]}; do

		echo $(date)
		echo "Mocore Queens - $q"
		./bin/fsp.out --size=${q} --lower_bound="queens" --mode="mcore" --scheduler="dynamic" --initial_depth=7 --lchunk=32 -nl1  >> results/mcore${q}.txt
		#./bin/fsp.out --size=$q --lower_bound="queens" --mode="improved" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=true --pgas=true --initial_depth=2 --second_depth=8 --mlchunk=1 --lchunk=1 -nl $((loc+1))
	done 

}


#tests1206
#testsGuided
testsStatic
#testsMcore

