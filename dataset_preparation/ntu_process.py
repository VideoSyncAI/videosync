# coding=utf-8
import os
import json
import cv2
from tqdm import tqdm
import pickle
import numpy as np


def main(split="train"):
    data_root = "/data/ntu"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    if split == "train":
        save_file = os.path.join(data_root, f"train.pkl")
    else:
        save_file = os.path.join(data_root, f"val.pkl")

    video_dir = os.path.join(data_root, "raw_videos")
    ntu_syn_dir = os.path.join(
        data_root, f"NTU-SYN/pose/{'train' if split == 'train' else 'test'}")

    labels = [{}, {}]
    num_frames = [{}, {}]
    event_ids = set()

    video_pair_ids = set()

    for root, _, files in os.walk(ntu_syn_dir):
        for i, file in enumerate(files):
            if not file.endswith('.json'):
                continue

            camera = file[5:8]
            if camera != '001':
                continue

            # Sample file name:
            # S001C002P001R002A007_rgb.avi.json
            # S: scene, C: camera, P: performer, R: replication, A: action

            event_id = file.split("_rgb")[0]
            event_ids.add(event_id)

            for i, camera_id in enumerate(['001', '002']):
                file = f'{file[0:5]}{camera_id}{file[8:]}'
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    labels[i][event_id] = data['category_id']
                    num_frames[i][event_id] = data['info']['num_frame']

    dataset = []

    for i, event_id in tqdm(enumerate(event_ids), total=len(event_ids)):
        data_dict = {"id": i, "name": event_id}
        
        # Pick shorter duration among two videos (while taking into account the frame offset),
        # So that end processed video pair have the same video duration.
        # This is need to prevent information leak from the difference in video duration.
        num_frame = min(num_frames[0][event_id] - labels[0][event_id], num_frames[1][event_id] - labels[1][event_id])
        for j, camera_id in enumerate(['001', '002']):
            video_id = f'{event_id[:5]}{camera_id}{event_id[8:]}'
            output_file = os.path.join(output_dir, video_id) + ".mp4"
            if not os.path.exists(output_file):
                video_path = os.path.join(video_dir, video_id)
                suffix = "_rgb.avi"

                if os.path.exists(video_path+suffix):
                    video_file = video_path+suffix
                else:
                    continue

                frame_offset = labels[j][event_id]
                frame_select_filter = f'select=gte(n\,{str(frame_offset)}),setpts=PTS-STARTPTS,'

                cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_file} -strict -2 -vf "{frame_select_filter}scale=224:224,setdar=1:1" -r 30 {output_file}'
                os.system(cmd)

            video = cv2.VideoCapture(output_file)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            data_dict[f"video_file_{j}"] = output_file
            data_dict[f"seq_len_{j}"] = int(
                video.get(cv2.CAP_PROP_FRAME_COUNT))
            data_dict[f"label_{j}"] = labels[j][event_id]

        dataset.append(data_dict)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of NTU dataset have been writen.")


if __name__ == '__main__':
    main(split="train")
    main(split="val")
