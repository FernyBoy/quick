#!/bin/bash

runpath='runs'
mkdir $runpath
echo "Storing results in $runpath"

networks_and_features() {
    python eam.py -n --runpath="$runpath" && \
    python eam.py -f --runpath="$runpath" && \
    echo "Creation of neural networks and generation of features is done!"
    if [ "$?" -ne 0 ]; then
        echo "ABORTING ON FAILURE AT CREATING NETWORKS OF GENERATING FEATURES"
        exit 1
    fi
}

# networks_and_features

for num_classes in 2 4 8 16 24 32 48 64; do
    python eam.py -e 1 --num-classes="$num_classes" --runpath="$runpath" && \
    python eam.py -e 2 --num-classes="$num_classes" --runpath="$runpath" && \
    # python eam.py -r --num-classes="$num_classes" --runpath="$runpath" && \
    echo "Done for $num_clases classes!"
    if [ "$?" -ne 0 ]; then
        echo "ABORTING ON FAILURE AT $num_classes classes"
	exit 1
    fi
done
