#!/bin/bash

command="$@"
docker run -it markub/rl-toolkit:cpu \
    bash -c "$command"
