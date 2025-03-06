# coding=utf-8
import os
import json
import cv2
from tqdm import tqdm
import pickle

num_to_class_str = {
    "1": "waving",
    "2": "shaking",
    "3": "patting",
    "4": "high-fiving",
    "5": "exchange",
    "6": "hugging"
}


def main(split="train"):
    data_root = "/home/username/koa_scratch/cvid"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    if split == "train":
        save_file = os.path.join(data_root, f"train.pkl")
    else:
        save_file = os.path.join(data_root, f"val.pkl")

    images_dir = os.path.join(data_root, "raw_images")
    cvid_syn_dir = os.path.join(
        data_root, f"CVID-SYN/pose")

    labels = [{}, {}]
    event_ids = set()

    video_pair_ids = set()

    for root, _, files in os.walk(cvid_syn_dir):
        for i, file in enumerate(files):
            if not file.endswith('.json'):
                continue

            # Let's arbitrarily choose odd class numbers as training and even class numbers as test set
            class_num = int(file[4])
            is_train_set = class_num in [1, 3, 5]
            if (split == "train" and not is_train_set) or (split == "val" and is_train_set):
                continue

            # Sample file name:
            # C1_V1_1_00.json
            # C1: cut point (one of five cut points for the same 120 frames video)
            # V1: class (one of six classes of actions)
            #   1: waving
            #   2: shaking
            #   3: patting
            #   4: high-fiving (we don't raw frames for this so we should ignore them)
            #   5: exchange
            #   6: hugging
            # Note that the above classes are a best guess at best because the authors were not clear about the file name conventions.
            #
            # 1: Camera number (1 of 2)
            # cc x00: index into x'th 120 frames video

            camera = file[6]
            if camera != '1':
                continue

            event_id = file
            event_ids.add(event_id)

            for i, camera_id in enumerate(['1', '2']):
                file = f'{file[0:6]}{camera_id}{file[7:]}'
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    labels[i][event_id] = data['category_id']

    dataset = []

    for i, event_id in tqdm(enumerate(event_ids), total=len(event_ids)):
        data_dict = {"id": i, "name": event_id}
        for j, camera_id in enumerate(['1', '2']):
            video_id = f'{event_id[:6]}{camera_id}{event_id[7:]}'
            output_file = os.path.join(output_dir, video_id) + ".mp4"
            if not os.path.exists(output_file):
                view_dir = 'View1' if camera_id == '1' else 'View2'
                action_dir = num_to_class_str[event_id[4]]

                images_path = os.path.join(images_dir, view_dir, action_dir)
                chunk_index = int(event_id[8:10])
                frame_offset = labels[j][event_id]
                start_frame = chunk_index * 120 + frame_offset

                if not os.path.exists(images_path):
                    continue

                cmd = f'ffmpeg -hide_banner -loglevel panic -y -start_number {start_frame} -i {images_path}/%04d.jpg -strict -2 -frames:v 120 -vf "scale=224:224,setdar=1:1" {output_file}'
                os.system(cmd)

            video = cv2.VideoCapture(output_file)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            data_dict[f"video_file_{j}"] = output_file
            data_dict[f"seq_len_{j}"] = num_frames
            data_dict[f"label_{j}"] = labels[j][event_id]

        if ("video_file_0" not in data_dict):
            print(f"skipping event_id: {event_id}")
            continue

        dataset.append(data_dict)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of CVID dataset have been writen.")


if __name__ == '__main__':
    main(split="train")
    main(split="val")
