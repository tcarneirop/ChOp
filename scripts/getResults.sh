for j in 1 2 4 8 16 32 59;do for i in $(seq 22 30);do echo `grep "Elapsed TOTAL" ${j}/COORDta${i}.txt|cut -d: -f2`;done;echo \"===\" $j;done


for j in 1 2 4 8 16 32 59;do for i in $(seq 22 30);do echo `grep "Ratio" teste|cut -d: -f2`;done;echo \"===\" $j;done


