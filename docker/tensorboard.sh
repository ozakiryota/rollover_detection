#!/bin/bash

if [ $# != 1 ]; then
	echo "Usage: ./tensorboard.sh LOG_DIR"
	exit 1
fi
log_dir=`basename $@`

xhost +

image="rollover_detection"
tag="latest"

docker run \
	-it \
	--rm \
	-e "DISPLAY" \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--gpus all \
	--net=host \
	-p 6006:6006 \
	-v $(pwd)/../log:/root/$image/log \
	$image:$tag \
	bash -c "tensorboard --logdir=/root/$image/log/$log_dir"