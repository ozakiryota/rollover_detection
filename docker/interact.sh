#!/bin/bash

xhost +

image="rollover_detection"
tag="latest"

docker run \
	-it \
	--rm \
	-e "DISPLAY" \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--gpus all \
	-v $HOME/dataset/rollover_detection:/root/dataset/rollover_detection \
	-v $(pwd)/../log:/root/$image/log \
	-v $(pwd)/../weights:/root/$image/weights \
	-v $(pwd)/../fig:/root/$image/fig \
	-v $(pwd)/../pyscr:/root/$image/pyscr \
	-v $(pwd)/../shell:/root/$image/shell \
	$image:$tag