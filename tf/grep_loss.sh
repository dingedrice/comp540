#!/bin/bash

for  i in 4 6;
do
	grep loss -A 1 cpu${i}*/sl*.out | awk 'NR%3==2 {print $1}' > ent_m${i}
	sed -i "s/\,//g" ent_m${i}
	grep loss -A 1 cpu${i}*/sl*.out | awk 'NR%3==2 {print $2}' > loss_m${i}
	sed -i "s/\,//g" loss_m${i}
done
