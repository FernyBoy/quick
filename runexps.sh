#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_classes>"
    exit 1
fi

num_classes="$1"
runpath="runs_$num_classes"
echo "Storing results in $runpath"
python eam.py -n --num-classes="$num_classes" --runpath="$runpath" && \
    python eam.py -f --num-classes="$num_classes" --runpath="$runpath" && \
    python eam.py -e 1 4 --num-classes="$num_classes" --runpath="$runpath" && \
    python eam.py -r --num-classes="$num_classes" --runpath="$runpath" && \
    python eam.py -d --num-classes="$num_classes" --runpath="$runpath"
