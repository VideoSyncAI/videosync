# coding=utf-8
import os
import json
import sys
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import subprocess


def get_video_frame_count(video_file_path):
    # Command to get the frame count using ffprobe
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames',
        '-of', 'json', video_file_path
    ]

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    # Parse the JSON output
    result_json = json.loads(result.stdout)

    # Extract the number of frames
    frame_count = int(result_json['streams'][0]['nb_read_frames'])

    return frame_count


def main(data_root, output_dir):
    # Create output video directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "processed_videos"), exist_ok=True)
    output_video_dir = os.path.join(output_dir, "processed_videos")

    labels = [{}, {}]
    num_frames = [{}, {}]

    video_name_roots = []
    video_name_prefixes = []
    video_name_prefixes_set = set()
    for root, _, files in os.walk(data_root):
        for i, file in enumerate(files):
            if not file.endswith('.mp4'):
                continue

            # "creamsoda_to_clear5_real_view_1.mp4" => "creamsoda_to_clear5_real_view_"
            prefix = file.split('.')[0][:-1]
            if prefix in video_name_prefixes_set:
                continue
            video_name_roots.append(root)
            video_name_prefixes.append(prefix)
            video_name_prefixes_set.add(prefix)
    dataset = []

    for i, (video_name_root, video_name_prefix) in tqdm(enumerate(zip(video_name_roots, video_name_prefixes)), total=len(video_name_prefixes)):
        data_dict = {"id": i, "name": video_name_prefix}

        input_video1_path = os.path.join(video_name_root, f"{video_name_prefix}0.mp4")
        input_video2_path = os.path.join(video_name_root, f"{video_name_prefix}1.mp4")

        output_video1_path = os.path.join(output_video_dir, f"{video_name_prefix}0.mp4")
        output_video2_path = os.path.join(output_video_dir, f"{video_name_prefix}1.mp4")

        if not os.path.exists(input_video1_path) or not os.path.exists(input_video2_path):
            print(f"Skipping the entry because we are missing some keys. name: {video_name_prefix}, data_dict.keys(): {data_dict.keys()}")
            continue

        # positive offset k means the first video starts at k'th frame and the second video starts at 0.
        # negative offset -k means the first video starts at 0'th frame and the second video starts at k'th frame
        frame_offset = np.random.randint(-30, 30)
        if frame_offset > 0:
            video1_start_frame = frame_offset
            video2_start_frame = 0
            # End time is the minimum of the two videos, while taking into account the offset
            video_duration = min(get_video_frame_count(input_video1_path) - video1_start_frame, get_video_frame_count(input_video2_path))
            video1_end_frame = video1_start_frame + video_duration
            video2_end_frame = video_duration   
        else:
            video1_start_frame = 0
            video2_start_frame = -1 * frame_offset
            video_duration = min(get_video_frame_count(input_video1_path), get_video_frame_count(input_video2_path) - video2_start_frame)
            video1_end_frame = video_duration
            video2_end_frame = video2_start_frame + video_duration

        data_dict["seq_len_0"] = video1_end_frame - video1_start_frame
        data_dict["seq_len_1"] = video2_end_frame - video2_start_frame
        data_dict["label_0"] = video1_start_frame
        data_dict["label_1"] = video2_start_frame

        data_dict["video_file_0"] = output_video1_path
        data_dict["video_file_1"] = output_video2_path

        cap1 = cv2.VideoCapture(input_video1_path)
        cap2 = cv2.VideoCapture(input_video2_path)

        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(output_video1_path, fourcc, fps1, (224, 224))
        out2 = cv2.VideoWriter(output_video2_path, fourcc, fps2, (224, 224))

        try:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, video1_start_frame)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, video2_start_frame)

            for i in range(video1_end_frame - video1_start_frame):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    print(f"Error reading frames for {video_name_prefix} at frame {i}, ret1: {ret1}, ret2: {ret2}, video1_start_frame: {video1_start_frame}, video2_start_frame: {video2_start_frame}, video1_end_frame: {video1_end_frame}, video2_end_frame: {video2_end_frame}")
                    return
            
                frame1_resized = cv2.resize(frame1, (224, 224))
                frame2_resized = cv2.resize(frame2, (224, 224))

                out1.write(frame1_resized)
                out2.write(frame2_resized)        
        except Exception as e:
            print(f"Error processing video pair: {e}")
            return
        finally:
            cap1.release()
            cap2.release()
            out1.release()
            out2.release()

        dataset.append(data_dict)

    for split in ["train", "val"]:
        # Save same file for train and val because while we only have validation dataset,
        # the evaluation runtime expects both train and val files.
        save_file = os.path.join(output_dir, f"{split}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"{len(dataset)} {split} samples of Pouring dataset have been writen.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python pouring_process.py <data_root> <output_dir>")
        sys.exit(1)

    data_root = sys.argv[1]
    output_dir = sys.argv[2]
    main(data_root, output_dir)
