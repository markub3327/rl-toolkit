#!/bin/bash

command="$@"
docker run -it --rm --network host --ipc=host \
    --mount src=$(pwd),target=/root/rl-toolkit,type=bind markub/rl-toolkit:cpu \
    bash -c "$command"