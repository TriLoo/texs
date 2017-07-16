#!/bin/sh

function copyfile(){
	echo "Prepare files for NC"
	touch $1/$1_tb.v
	head -n 5 ./OR2/OR2_tb.v > $1/$1_tb.v
	mv $1.v $1
	cp ./OR2/filelist.v $1
	cp ./OR2/ncverilog.options $1
	cp ./OR2/run* $1
}
if [ ! -d "$1" ]; then
	mkdir "$1"
	echo "$1 is created"
	copyfile "$1"
else
	echo "$1 already existed"
	copyfile "$1"
fi
