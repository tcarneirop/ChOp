import os



locales = [59,32,16,8,4]

oitoquatro = [8,4]

vinteum = [59,32,16,8,4]

small = [2,1]

one = [1]

cnove = [59]

#27 may

for loc in locales:
        for instance in range(22,31):
                os.system("date")
                os.system("echo OLD PARAMETERS Johnson - %d locales, ta%d " % (loc, instance))
                os.system("./fsp.out  --instance=%d --coordinated=false --pgas=true --lower_bound=\"johnson\" -nl %d  >> johnson/%d/OLDta%d.txt" % (instance,loc,loc,instance))


for loc in locales:
        for instance in range(22,31):
                os.system("date")
                os.system("echo OLD PARAMETERS Simple - %d locales, ta%d " % (loc, instance))
                os.system("./fsp.out  --instance=%d --coordinated=false --pgas=true --lower_bound=\"simple\" -nl %d  >> simple/%d/OLDta%d.txt" % (instance,loc,loc,instance))

