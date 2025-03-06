# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import wandb

import os
import math
import torch
import pprint
import numpy as np
from tqdm import tqdm
import utils.logging as logging
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import utils.distributed as du
from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer
from datasets import construct_dataloader
from evaluation import get_tasks
from visualize_alignment import create_video, create_single_video, create_multiple_video
from visualize_retrieval import create_retrieval_video

logger = logging.get_logger(__name__)


def get_embeddings_dataset(cfg, model, data_loader):
    """Get embeddings from a one epoch iterator."""
    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS
    embs_list = []
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    model.eval()
    with torch.no_grad():
        for video, frame_label, seq_len, chosen_steps, video_masks, names in data_loader:
            assert video.size(0) == 1  # batch_size==1

            assert video.size(1) == frame_label.size(1) == int(seq_len.item())
            embs = []
            seq_len = seq_len.item()
            num_batches = int(math.ceil(float(seq_len)/max_frames_per_batch))
            frames_per_batch = int(math.ceil(float(seq_len)/num_batches))
            for i in range(num_batches):
                curr_idx = i * frames_per_batch
                num_steps = min(seq_len - curr_idx, frames_per_batch)
                steps = torch.arange(curr_idx, curr_idx+num_steps)
                if num_contexts != 1:
                    # Get multiple context steps depending on config at selected steps.
                    context_stride = cfg.DATA.CONTEXT_STRIDE
                    steps = steps.view(-1, 1) + context_stride * \
                        torch.arange(-(num_contexts-1), 1).view(1, -1)
                steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
                curr_data = video[:, steps]
                if cfg.USE_AMP:
                    with torch.cuda.amp.autocast():
                        emb_feats = model(curr_data, num_steps)
                else:
                    emb_feats = model(curr_data, num_steps)
                embs.append(emb_feats[0].cpu())
            valid = (frame_label[0] >= 0)
            embs = torch.cat(embs, dim=0)
            embs_list.append(embs[valid].numpy())
            frame_labels_list.append(frame_label[0][valid].cpu().numpy())
            seq_lens_list.append(seq_len)
            input_lens_list.append(len(video[0]))
            steps_list.append(chosen_steps[0].cpu().numpy())
            names_list.append(names[0])

        dataset = {'embs': embs_list,
                   'labels': frame_labels_list,
                   'seq_lens': seq_lens_list,
                   'input_lens': input_lens_list,
                   'steps': steps_list,
                   'names': names_list}

        logger.info(f"embeddings_dataset size: {len(dataset['embs'])}")
    return dataset


def evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader,
                  iterator_tasks, embedding_tasks, cur_epoch, summary_writer):
    """Evaluate learnt embeddings on downstream tasks."""
    from evaluation.sync_offset import SyncOffset
    from algos import get_algo

    sync_offset = SyncOffset(cfg)
    algo = get_algo(cfg)

    sync_offset.evaluate(
                model, val_loader, val_emb_loader, cur_epoch, summary_writer, sample=False, cur_iter=0, algo=algo)


@record
def evaluate():
    """Evaluate embeddings."""
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args

    wandb.init(project="videosync_scl", sync_tensorboard=True, config=cfg)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    du.init_distributed_training(cfg)
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    # Setup summary writer.
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'eval_logs'))

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
    train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    val_loader, [val_emb_loader] = construct_dataloader(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader,
                  iterator_tasks, embedding_tasks, start_epoch, summary_writer)


if __name__ == '__main__':
    wandb.require("service")
    evaluate()
    wandb.finish()
