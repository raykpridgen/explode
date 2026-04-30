#!/bin/bash

cd ../data/cyl/

for i in $(seq -f "%05g" 1600 2100)
do
    wget -r -np -nd --spider \
    https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full/id$i/ 2>&1 \
    | grep '^--' | awk '{print $3}'
done > urls.txt

echo "DOWNLOAD START"

aria2c -i urls.txt -j 8 -x 8 -s 8 --continue=true --max-tries=0
