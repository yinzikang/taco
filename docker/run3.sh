#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=single_container isaacgym_share /bin/bash
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/w/Documents/IsaacGym:/isaacgym -e DISPLAY=$DISPLAY --network=host --gpus=all --name=single_container_test isaacgym_share /bin/bash
	xhost -
fi
