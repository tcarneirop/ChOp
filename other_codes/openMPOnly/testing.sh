for i in {0..250}; do ./a.out 14 > a ; grep "$(head -c 118 a)" 14out  && echo "String found in file" || echo "String not found in file"; sleep 0.8;done
