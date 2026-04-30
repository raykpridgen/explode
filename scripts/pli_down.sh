#!/bin/bash

cd ../data/cyl/

for i in $(seq -f "%05g" 1600 1800)
do
    wget -r -np -nd --spider \
    https://oceans11.lanl.gov/heat/pli/pli240420/id$i/ 2>&1 \
    | grep '^--' | awk '{print $3}'
done > urls.txt

echo "DOWNLOAD START"

aria2c -i urls.txt -j 8 -x 8 -s 8 --continue=true --max-tries=0
