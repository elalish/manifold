#!/bin/bash

# Go to manifold/extras directory and run the command as `./run.sh {path_to_dataset_folder} {name_of_csv} {implementation(Hull,Hull_CGAL)}`
# example ./run.sh ./Thingi10K/raw_meshes/ Hull4.csv Hull

# Checking if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output.csv> <Implementation>"
    exit 1
fi

EXECUTABLE="../build/extras/testHullPerformance" 
INPUT_FOLDER=$1
OUTPUT_CSV=$2
IMPLEMENTATION=$3
TIME_LIMIT=10m # time limit in minutes
RAM_LIMIT=6000  # Memory limit in MB

# Initializing the headers
echo "Filename,VolManifold,VolHull,AreaManifold,AreaHull,ManifoldTri,HullTri,Time,Status," > $OUTPUT_CSV

# Iterate over all files in the input folder
for INPUT_FILE in "$INPUT_FOLDER"/*; do
    FILE_NAME=$(basename "$INPUT_FILE")

    # Run the EXECUTABLE with the specified argument, time limit, and used to capture the output
    OUTPUT=$(ulimit -v $((RAM_LIMIT * 1024)); timeout $TIME_LIMIT  $EXECUTABLE "Input" "$IMPLEMENTATION" "0" "$INPUT_FILE" 2>&1)
    STATUS=$?

    # Checking if the EXECUTABLE timed out
    if [ $STATUS -eq 124 ]; then
        STATUS="Timeout"
    elif [ $STATUS -ne 0 ]; then
        STATUS="Error"
    else
        STATUS="Success"
    fi

    # Adding the result to the output file
    echo "\"$FILE_NAME\",$OUTPUT,\"$STATUS\"" >> $OUTPUT_CSV
done
