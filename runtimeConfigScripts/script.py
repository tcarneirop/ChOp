import os

for x in range(1,15):
	os.system("echo EXECUTION %s" % (x) )
	os.system("bin/./fsp.out  --instance=12 --mode=\"improved\" --mlchunk=8 --coordinated=false --pgas=true --lower_bound=\"johnson\" --scheduler=\"dynamic\" --initial_depth=4 --second_depth=7  -nl 8")
