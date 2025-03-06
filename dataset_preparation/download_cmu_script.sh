#!/bin/bash

# List of datasets
datasets=(
    "171204_pose1"
    "171204_pose2"
    "171204_pose3"
    "171204_pose4"
    "171204_pose5" 
    "171204_pose6"
    "171026_pose1"
    "171026_pose2"
    "171026_pose3"
)

# Base URLs and paths
base_url="http://domedb.perception.cs.cmu.edu/webdata/dataset"
base_dir="/data/cmu/pose_dataset"

# Video files to download for each dataset
video_files=(
    "hd_00_00.mp4"
    "hd_00_03.mp4"
    "hd_00_06.mp4"
    "hd_00_09.mp4"
    "hd_00_10.mp4"
    "hd_00_11.mp4"
    "hd_00_15.mp4"
    "hd_00_20.mp4"
    "hd_00_21.mp4"
    "hd_00_24.mp4"
    "hd_00_27.mp4"
    "hd_00_30.mp4"
)

# Download each dataset
for dataset in "${datasets[@]}"; do
    # Create output directory
    output_dir="${base_dir}/${dataset}/hdVideos"
    mkdir -p "$output_dir"
    
    # Download each video file
    for video in "${video_files[@]}"; do
        url="${base_url}/${dataset}/videos/hd_shared_crf20/${video}"
        output_file="${output_dir}/${video}"
        
        # Check if file already exists
        if [ -f "$output_file" ]; then
            echo "File already exists: $output_file"
        else
            echo "Downloading: $url"
            wget -P "$output_dir" "$url"
        fi
    done
done
