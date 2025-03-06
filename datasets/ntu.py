# coding=utf-8
import os
import math
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

import utils.logging as logging
from datasets.data_augment import create_data_augment, create_ssl_data_augment

from logging import INFO
logger = logging.get_logger(__name__)
logger.setLevel(INFO)

prefix = ""

class Ntu(torch.utils.data.Dataset):
    def __init__(self, cfg, split, mode="auto", sample_all=False):
        assert split in ["train", "val"]
        self.cfg = cfg
        self.split = split
        if mode == "auto":
            self.mode = "train" if self.split == "train" else "eval"
        else:
            self.mode = mode
        self.sample_all = sample_all
        self.num_contexts = cfg.DATA.NUM_CONTEXTS

        self.dataset_name = cfg.DATASETS[0]

        self.train_dataset = os.path.join(
            cfg.args.workdir, self.dataset_name, f"train.pkl")
        self.val_dataset = os.path.join(
            cfg.args.workdir, self.dataset_name, f"val.pkl")

        if self.split == "train":
            with open(self.train_dataset, 'rb') as f:
                dataset = pickle.load(f)
        else:
            with open(self.val_dataset, 'rb') as f:
                dataset = pickle.load(f)

        self.dataset = []
        self.error_videos = []
        self.dataset = dataset

        self.num_frames = cfg.TRAIN.NUM_FRAMES
        if self.cfg.SSL and self.mode == "train":
            self.data_preprocess = create_ssl_data_augment(cfg, augment=True)
        elif self.mode == "train":
            self.data_preprocess = create_data_augment(cfg, augment=True)
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.cfg.SSL and (self.mode == "train" or not self.sample_all):
            # return self.get_supervised_training_item(index)
            return self.get_training_item(index)

        videos = []
        labels = []
        seq_lens = []
        chosen_steps = []
        video_masks = []
        names = []

        # First, get both video lengths to determine the shorter duration
        video_lengths = []
        for i in range(2):
            video_file = os.path.join(
                self.cfg.args.workdir, self.dataset_name, self.dataset[index][f"video_file_{i}"][len(prefix):])
            video, _, _ = read_video(video_file, pts_unit='sec')
            video_lengths.append(len(video))

        min_length = min(video_lengths)

        for i, camera_id in enumerate(['001', '002']):
            name = self.dataset[index][f"video_file_{i}"].split(
                "/")[-1].split(".")[0]
            video_file = os.path.join(
                self.cfg.args.workdir, self.dataset_name, self.dataset[index][f"video_file_{i}"][len(prefix):])
            video, _, info = read_video(video_file, pts_unit='sec')
            # Crop video to shorter length
            video = video[:min_length]
            seq_len = len(video)
            if seq_len == 0:
                print('seq_len is 0', video_file)
            # T H W C -> T C H W, [0,1] tensor
            video = video.permute(0, 3, 1, 2).float() / 255.0

            # Use random signal instead of actual frame
            # if i == 1:
            # video = torch.rand(video.shape)

            # Crop video to first X frames
            # video = video[:300]
            # seq_len = len(video)  # Update seq_len to new length

            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            video = video[steps.long()]
            video = self.data_preprocess(video)
            label = torch.full(
                (1, ), self.dataset[index][f"label_{i}"], dtype=torch.int32)
            seq_len = len(steps)
            chosen_step = steps.clone()
            video_mask = torch.ones(seq_len)

            videos.append(video)
            labels.append(label)
            seq_lens.append(seq_len)
            chosen_steps.append(chosen_step)
            video_masks.append(video_mask)
            names.append(name)

        # videos = torch.stack(videos, dim=0)
        # labels = torch.stack(labels, dim=0)
        # seq_lens = torch.tensor(seq_lens)
        # chosen_steps = torch.stack(chosen_steps, dim=0)
        # video_masks = torch.stack(video_masks, dim=0)

        return videos, labels, seq_lens, chosen_steps, video_masks, names

    def get_training_item(self, index):
        # XXX: Take the first video from the pair of videos. We will figure out how to utilize the second videos later.
        name = self.dataset[index][f"video_file_0"].split(
            "/")[-1].split(".")[0]


        video_file = os.path.join(
            self.cfg.args.workdir, self.dataset_name, self.dataset[index][f"video_file_0"][len(prefix):])
        video, _, info = read_video(video_file, pts_unit='sec')
        # T H W C -> T C H W, [0,1] tensor
        video = video.permute(0, 3, 1, 2).float() / 255.0
        seq_len = len(video)
        frame_label = -1 * torch.ones(seq_len)

        names = [name, name]
        steps_0, chosen_step_0, video_mask0 = self.sample_frames(
            seq_len, self.num_frames)

        logger.debug(
            f'name: {name}, video[steps_0.long()].shape: {video[steps_0.long()].shape}, video[steps_0.long()].dtype: {video[steps_0.long()].dtype}')

        view_0 = self.data_preprocess(video[steps_0.long()])
        label_0 = frame_label[chosen_step_0.long()]
        steps_1, chosen_step_1, video_mask1 = self.sample_frames(
            seq_len, self.num_frames, pre_steps=steps_0)
        view_1 = self.data_preprocess(video[steps_1.long()])
        label_1 = frame_label[chosen_step_1.long()]
        videos = torch.stack([view_0, view_1], dim=0)
        labels = torch.stack([label_0, label_1], dim=0)
        seq_lens = torch.tensor([seq_len, seq_len])
        chosen_steps = torch.stack([chosen_step_0, chosen_step_1], dim=0)
        video_mask = torch.stack([video_mask0, video_mask1], dim=0)
        return videos, labels, seq_lens, chosen_steps, video_mask, names

    def get_supervised_training_item(self, index):
        videos = []
        labels = []
        seq_lens = []
        chosen_steps = []
        video_masks = []
        names = []

        # Let's not random crop and make other transformations for supervised learning.
        data_preprocess = create_data_augment(self.cfg, augment=False)

        steps_0 = None
        for i, camera_id in enumerate(['001', '002']):
            name = self.dataset[index][f"video_file_{i}"].split(
                "/")[-1].split(".")[0]
            video_file = os.path.join(
                self.cfg.args.workdir, self.dataset_name, self.dataset[index][f"video_file_{i}"][len(prefix):])
            video, _, info = read_video(video_file, pts_unit='sec')
            # T H W C -> T C H W, [0,1] tensor
            video = video.permute(0, 3, 1, 2).float() / 255.0
            seq_len = len(video)
            frame_label = -1 * torch.ones(seq_len)

            steps, chosen_step, video_mask = self.sample_frames(
                seq_len, self.num_frames, pre_steps=steps_0)
            view = data_preprocess(video[steps.long()])
            label = frame_label[chosen_step.long()]

            if i == 0:
                steps_0 = steps.clone()

            frame_offset_label = torch.full(
                (1, ), self.dataset[index][f"label_{i}"], dtype=torch.int32).item()
            if frame_offset_label != 0:
                chosen_step = chosen_step + frame_offset_label

            videos.append(view)
            labels.append(label)
            seq_lens.append(seq_len)
            chosen_steps.append(chosen_step)
            video_masks.append(video_mask)
            names.append(name)

        videos = torch.stack(videos, dim=0)
        labels = torch.stack(labels, dim=0)
        seq_lens = torch.tensor(seq_lens)
        chosen_steps = torch.stack(chosen_steps, dim=0)
        video_masks = torch.stack(video_masks, dim=0)
        return videos, labels, seq_lens, chosen_steps, video_masks, names

    def sample_frames(self, seq_len, num_frames, pre_steps=None):
        # When dealing with very long videos we can choose to sub-sample to fit
        # data in memory. But be aware this also evaluates over a subset of frames.
        # Subsampling the validation set videos when reporting performance is not
        # recommended.
        sampling_strategy = self.cfg.DATA.SAMPLING_STRATEGY
        pre_offset = min(pre_steps) if pre_steps is not None else None

        if sampling_strategy == 'offset_uniform':
            # Sample a random offset less than a provided max offset. Among all frames
            # higher than the chosen offset, randomly sample num_frames
            if seq_len >= num_frames:
                # Returns a random permutation of integers from 0 to n - 1.
                steps = torch.randperm(seq_len)
                steps = torch.sort(steps[:num_frames])[0]
            else:
                steps = torch.arange(0, num_frames)
        elif sampling_strategy == 'time_augment':
            num_valid = min(seq_len, num_frames)
            expand_ratio = np.random.uniform(
                low=1.0, high=self.cfg.DATA.SAMPLING_REGION) if self.cfg.DATA.SAMPLING_REGION > 1 else 1.0

            block_size = math.ceil(expand_ratio*num_valid)
            if pre_steps is not None and self.cfg.DATA.CONSISTENT_OFFSET != 0:
                shift = int((1-self.cfg.DATA.CONSISTENT_OFFSET)*num_valid)
                offset = np.random.randint(low=max(0, min(
                    seq_len-block_size, pre_offset-shift)), high=max(1, min(seq_len-block_size+1, pre_offset+shift+1)))
            else:
                offset = np.random.randint(
                    low=0, high=max(seq_len-block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames-num_valid),
                              "constant", seq_len)
        else:
            raise ValueError('Sampling strategy %s is unknown. Supported values are '
                             'stride, offset_uniform .' % sampling_strategy)

        if 'tcn' in self.cfg.TRAINING_ALGO:
            pos_window = self.cfg.TCN.POSITIVE_WINDOW
            pos_steps = steps + torch.randint(-pos_window, 0, steps.size())
            steps = torch.stack([steps, pos_steps], dim=0)
            steps = steps.transpose(0, 1).contiguous().view(-1)
            num_frames = num_frames*2

        video_mask = torch.ones(num_frames)
        video_mask[steps < 0] = 0
        video_mask[steps >= seq_len] = 0
        # Store chosen indices.
        chosen_steps = torch.clamp(steps.clone(), 0, seq_len - 1)
        if self.num_contexts == 1:
            steps = chosen_steps
        else:
            # Get multiple context steps depending on config at selected steps.
            context_stride = self.cfg.DATA.CONTEXT_STRIDE
            steps = steps.view(-1, 1) + context_stride * \
                torch.arange(-(self.num_contexts-1), 1).view(1, -1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)

        return steps, chosen_steps, video_mask
