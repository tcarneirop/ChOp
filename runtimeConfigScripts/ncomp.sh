#!/bin/sh

ncompi=$(cat ncompilations)

var=$(($ncompi+1))


echo "### Number of compilations: " $var "####"
echo $var > ncompilations 