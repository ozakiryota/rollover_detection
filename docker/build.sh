#!/bin/bash

image="rollover_detection"
tag="latest"

docker build . \
    -t $image:$tag \
    --build-arg CACHEBUST=$(date +%s)