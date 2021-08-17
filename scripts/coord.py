import os


locales = [32,16,8,4]

small = [2]

one = [1]


	
for loc in locales:
	for instance in range(30,31):
		os.system("date")
		os.system("echo NEW PARAMETERS COORDINATED PGAS Johnson - %d locales, ta%d " % (loc, instance))
		os.system("./fsp.out  --instance=%d --mlchunk=32 --slchunk=4 --coordinated=true --pgas=true --lower_bound=\"johnson\" -nl %d  >> johnson/%d/COORDPGAS3204ta%d.txt" % (instance,loc,loc,instance))

for loc in locales:
	for instance in range(30,31):
		os.system("date")
		os.system("echo NEW PARAMETERS COORDINATED PGAS Simple - %d locales, ta%d " % (loc, instance))
		os.system("./fsp.out  --instance=%d --mlchunk=8 --slchunk=2 --coordinated=true --pgas=true --lower_bound=\"simple\" -nl %d  >> simple/%d/COORDPGAS0802ta%d.txt" % (instance,loc,loc,instance))



for loc in small:
	for instance in range(30,31):
		os.system("date")
		os.system("echo NEW PARAMETERS PGAS Johnson - %d locales, ta%d " % (loc, instance))
		os.system("./fsp.out  --instance=%d --mlchunk=32 --slchunk=4 --coordinated=true --pgas=true --lower_bound=\"johnson\" -nl %d  >> johnson/%d/COORDPGAS3204ta%d.txt" % (instance,loc,loc,instance))

for loc in small:
	for instance in range(30,31):
		os.system("date")
		os.system("echo NEW PARAMETERS PGAS Simple - %d locales, ta%d " % (loc, instance))
		os.system("./fsp.out  --instance=%d --mlchunk=8 --slchunk=2 --coordinated=true --pgas=true --lower_bound=\"simple\" -nl %d  >> simple/%d/COORDPGAS0802ta%d.txt" % (instance,loc,loc,instance))
