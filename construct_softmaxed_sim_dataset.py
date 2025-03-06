# coding=utf-8

import os
import torch
import pprint
import numpy as np
from tqdm import tqdm
import utils.logging as logging

from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, load_checkpoint
from utils.optimizer import construct_optimizer
from datasets import construct_dataloader

from evaluation.sync_offset import SyncOffset, get_similarity
from torch.nn import functional as Fun
from torch.distributed.elastic.multiprocessing.errors import record
import utils.distributed as du
import random

logger = logging.get_logger(__name__)


def save_similarity_and_labels(cfg, model, loader, dataset_type):
    model.eval()  # Set model to evaluation mode

    all_similarity_matrices = []
    all_labels = []

    sync_offset_calculator = SyncOffset(cfg)

    for videos, labels, seq_lens, chosen_steps, video_masks, names in tqdm(loader, desc=f"Processing {dataset_type} data"):
        embs = []
        for i in [0, 1]:
            video = videos[i]
            seq_len = seq_lens[i]
            chosen_step = chosen_steps[i]
            video_mask = video_masks[i]
            name = names[i]

            emb = sync_offset_calculator.get_embs(
                model, video, labels[i], seq_len, chosen_step, video_mask, name)
            embs.append(emb)

        similarity_matrix = get_similarity(
            torch.tensor(embs[0]).cuda(), torch.tensor(embs[1]).cuda())
        softmaxed_similarity_matrix = Fun.softmax(
            similarity_matrix, dim=1).cpu().numpy()

        all_similarity_matrices.append(softmaxed_similarity_matrix)
        all_labels.append((labels[0] - labels[1]).item())

    # Save data only from the main process
    if torch.distributed.get_rank() == 0:
        prefix = cfg.args.dataset_prefix
        np.save(f'{prefix}_{dataset_type}_softmaxed_sim_12.npy',
                np.array(all_similarity_matrices))
        np.save(f'{prefix}_{dataset_type}_softmaxed_sim_12_labels.npy',
                np.array(all_labels))


def gather_data_from_all_processes(data):
    data_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(data_list, data)
    data = [item for sublist in data_list for item in sublist]  # Flatten list
    return data


@record
def main():
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=False)
    optimizer = construct_optimizer(model, cfg)
    start_epoch = load_checkpoint(cfg, model, optimizer)

    # Setup Dataset Iterators from train and val datasets.
    for split in ["train", "val"]:
        _, [emb_loader] = construct_dataloader(cfg, split)

        with torch.no_grad():
            save_similarity_and_labels(cfg, model, emb_loader, split)


if __name__ == '__main__':
    main()
