import os


method = [
"0",
"1"
]


sizes = [
"10"
]

bigSizes = [
"18",
"19"
]


geant = [
"20"
]



Locales = [
"1",
"2",
"4",
"8",
"16",
"32"
]



bigLocales = [
"32",
"16",
"8",
"4",
"2",
"1"]



for nl in bigLocales:
	for s in sizes:
		os.system("time -p \( /usr/bin/mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey \"0x8100\" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \)  2>> times/dynamic/%s/%s.txt" % (nl,s,nl,s))


#for nl in bigLocales:
#	for s in sizes:
#		os.system("time \(mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \) 2> experiments/times/static/%s/%s.txt" % (nl,s,nl,s))




#for nl in bigLocales:
#	for s in bigSizes:
#		os.system("time \(mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \) 2> experiments/times/dynamic/%s/%s.txt" % (nl,s,nl,s))


#for nl in bigLocales:
#	for s in bigSizes:
#		os.system("time \(mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \) 2> experiments/times/static/%s/%s.txt" % (nl,s,nl,s))





#for nl in bigLocales:
#	for s in geant:
#		os.system("time \(mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \) 2> experiments/times/dynamic/%s/%s.txt" % (nl,s,nl,s))


#for nl in bigLocales:
#	for s in geant:
#		os.system("time \(mpirun --bind-to none --map-by ppr:1:node --mca btl_openib_pkey "0x8100" -np %s -machinefile $OAR_NODEFILE queens %s 4 1 \) 2> experiments/times/static/%s/%s.txt" % (nl,s,nl,s))



