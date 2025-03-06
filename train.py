# coding=utf-8

import wandb

import os
import sys
import pprint
import torch
import random
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record
import torch.autograd.profiler as profiler
import signal
import threading

import utils.distributed as du
import utils.logging as logging
from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler, get_lr
from datasets import construct_dataloader, unnorm
from algos import get_algo
from evaluation import get_tasks

logger = logging.get_logger(__name__)

prof = None
profiler_output_path = None


def signal_handler(sig, frame):
    global prof, profiler_output_path

    if prof is None:
        return

    print('Signal received, exporting profiler data...')
    prof.export_chrome_trace(profiler_output_path)
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, summary_writer, val_loader, val_emb_loader, iterator_tasks):
    global prof

    model.train()
    optimizer.zero_grad()
    data_size = len(train_loader)
    # DistributedSampler shuffle based on epoch and seed
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(cur_epoch)
        logger.info(f"update the training sampler to epoch {cur_epoch}")
    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(cur_epoch)
        logger.info(f"update the training batch sampler to epoch {cur_epoch}")
    total_loss = {}

    logger.info(
        f'epoch: {cur_epoch}, LR: {get_lr(optimizer)}, scheduler.last_epoch: {scheduler.last_epoch}')

    from evaluation.sync_offset import SyncOffset
    sync_offset = SyncOffset(cfg)

    if du.is_root_proc():
        train_loader = tqdm(train_loader, total=len(train_loader))
    for cur_iter, (videos, _labels, seq_lens, chosen_steps, video_masks, names) in enumerate(train_loader):
        logger.debug(f'cur_iter: {cur_iter}, name: {names[0]}')

        if cur_iter % 500 == 0:
            sync_offset_res = sync_offset.evaluate(
                model, val_loader, val_emb_loader, cur_epoch, summary_writer, sample=True, cur_iter=cur_iter, algo=algo)

            model.train()

        if cur_epoch == 32 and cur_iter == 0:
            logger.info('turning on profiling')
            prof = profiler.profile(use_cuda=True)
            prof.__enter__()

        optimizer.zero_grad()
        if cfg.USE_AMP:
            torch.autograd.set_detect_anomaly(True)
            scaler = algo.scaler
            with torch.cuda.amp.autocast():
                if cfg.TRAINING_ALGO == 'classification':
                    loss_dict = algo.compute_loss(
                        model, videos, _labels, seq_lens, chosen_steps, video_masks)
                else:
                    loss_dict = algo.compute_loss(
                        model, videos, seq_lens, chosen_steps, video_masks)
            loss = loss_dict["loss"]

            logger.info(
                f'cur_iter: {cur_iter}, loss: {loss_dict["loss"].item()}, pid:{os.getpid()}')

            scaler.scale(loss).backward()
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            torch.autograd.set_detect_anomaly(True)
            if cfg.TRAINING_ALGO == 'classification':
                loss_dict = algo.compute_loss(
                    model, videos, _labels, seq_lens, chosen_steps, video_masks)
            else:
                loss_dict = algo.compute_loss(
                    model, videos, seq_lens, chosen_steps, video_masks)
            loss = loss_dict["loss"]

            def target():
                # Perform the backward pass.
                loss.backward()

            # Timeout after 60 seconds
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(60)
            if thread.is_alive():
                logger.error(
                    f'backward pass timed out. cur_epoch: {cur_epoch}, cur_iter: {cur_iter}, name: {names[0]}')

                if cur_epoch == 32 and cur_iter == 0:
                    logger.info('turning off profiling')
                    prof.__exit__(None, None, None)
                    prof.export_chrome_trace(profiler_output_path)

                if prof is None:
                    sys.exit(0)
                print('Signal received, exporting profiler data...')
                prof.export_chrome_trace(profiler_output_path)
                sys.exit(0)

            # Update the parameters.
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            optimizer.step()

        for key in loss_dict:
            loss_dict[key][torch.isnan(loss_dict[key])] = 0
            if key not in total_loss:
                total_loss[key] = 0
            total_loss[key] += du.all_reduce([loss_dict[key]]
                                             )[0].item() / data_size

        if cur_epoch == 32 and cur_iter == 0:
            logger.info('turning off profiling')
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(profiler_output_path)

        if du.is_root_proc() and cur_iter % cfg.LOGGING.REPORT_INTERVAL == 0:
            logger.info(
                f"iter {data_size * cur_epoch + cur_iter}, training loss: {loss.item():.3f}")

        wandb.log({
            "loss": loss,
        })

        du.synchronize()

    summary_writer.add_scalar('train/learning_rate',
                              get_lr(optimizer)[0], cur_epoch)
    for key in total_loss:
        summary_writer.add_scalar(f'train/{key}', total_loss[key], cur_epoch)
    logger.info("epoch {}, train loss: {:.3f}".format(
        cur_epoch, total_loss["loss"]))

    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()


def val(cfg, val_loader, model, algo, cur_epoch, summary_writer):
    model.eval()
    data_size = len(val_loader)
    total_loss = {}

    with torch.no_grad():
        for cur_iter, (videos, labels, seq_lens, chosen_steps, video_masks, names) in enumerate(val_loader):
            if cfg.USE_AMP:
                with torch.cuda.amp.autocast():
                    if cfg.TRAINING_ALGO == 'classification':
                        loss_dict = algo.compute_loss(
                            model, videos, labels, seq_lens, chosen_steps, video_masks, training=False)
                    else:
                        loss_dict = algo.compute_loss(
                            model, videos, seq_lens, chosen_steps, video_masks, training=False)
            else:
                if cfg.TRAINING_ALGO == 'classification':
                    loss_dict = algo.compute_loss(
                        model, videos, labels, seq_lens, chosen_steps, video_masks, training=False)
                else:
                    loss_dict = algo.compute_loss(
                        model, videos, seq_lens, chosen_steps, video_masks, training=False)

            for key in loss_dict:
                loss_dict[key][torch.isnan(loss_dict[key])] = 0
                if key not in total_loss:
                    total_loss[key] = 0
                total_loss[key] += du.all_reduce([loss_dict[key]]
                                                 )[0].item() / data_size

        if cfg.NUM_GPUS == 1:
            print(names)
            visual_video = videos[0]
            if cfg.SSL:
                for i, v in enumerate(visual_video):
                    summary_writer.add_video(
                        f'{names}_view{i}', unnorm(v[::2]).unsqueeze(0), 0, fps=4)
            else:
                summary_writer.add_video(f'{names}', unnorm(
                    visual_video[::2]).unsqueeze(0), 0, fps=4)

    for key in total_loss:
        summary_writer.add_scalar(f'val/{key}', total_loss[key], cur_epoch)
    logger.info("epoch {}, val loss: {:.3f}".format(
        cur_epoch, total_loss["loss"]))


@record
def main():
    global profiler_output_path

    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()  # num_gpus_per_machine

    args.world_size = int(os.getenv('WORLD_SIZE'))  # total_gpus
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
        args.rank = args.local_rank
    else:
        args.node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.rank = args.node_rank * torch.cuda.device_count() + args.local_rank
    logger.info(f'Node info: rank {args.rank} of world size {args.world_size}')
    cfg.args = args

    profiler_output_path = f'/home/username/koa_scratch/profiler_log_{args.local_rank}.json'

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    wandb.init(project="videosync_scl", sync_tensorboard=True, config=cfg)

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    def check_gradients(module):
        for name, param in module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.warn(f"NaN gradient in {name}")

    model.register_full_backward_hook(
        lambda module, grad_input, grad_output: check_gradients(model))

    optimizer = construct_optimizer(model, cfg)
    algo = get_algo(cfg)

    # Setup Dataset Iterators from train and val datasets.
    train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    val_loader, [val_emb_loader] = construct_dataloader(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    if cfg.USE_AMP:
        algo.scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")

    """Trains model and evaluates on relevant downstream tasks."""
    start_epoch = load_checkpoint(cfg, model, optimizer) + 1
    cfg.TRAIN.MAX_ITERS = cfg.TRAIN.MAX_EPOCHS * len(train_loader)
    scheduler = construct_scheduler(optimizer, cfg)

    from evaluation.sync_offset import SyncOffset
    sync_offset = SyncOffset(cfg)

    for cur_epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
        logger.info(
            f"Traning epoch {cur_epoch}/{cfg.TRAIN.MAX_EPOCHS}, {len(train_loader)} iters each epoch")
        train(cfg, train_loader, model, optimizer, scheduler, algo,
              cur_epoch, summary_writer, val_loader, val_emb_loader, iterator_tasks)
        logger.info(f'done training cur_epoch: {cur_epoch}')
        if (cur_epoch+1) % cfg.EVAL.VAL_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1:
            # val(cfg, val_loader, model, algo, cur_epoch, summary_writer)
            if cfg.DATASETS[0] == "finegym":
                from evaluate_finegym import evaluate_once
                evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader,
                              iterator_tasks, embedding_tasks, cur_epoch, summary_writer)
            else:
                sync_offset_res = sync_offset.evaluate(
                    model, val_loader, val_emb_loader, cur_epoch, summary_writer, sample=False, cur_iter=0, algo=algo)
                model.train()
        if du.is_root_proc() and ((cur_epoch+1) % cfg.CHECKPOINT.SAVE_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1):
            logger.info('in root_proc, running save_checkpoint')
            save_checkpoint(cfg, model, optimizer, cur_epoch)
            logger.info('in root_proc, done running save_checkpoint')
        logger.info('running du.synchronize')
        du.synchronize()
        logger.info('done running du.synchronize')

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    wandb.require("service")
    main()
    wandb.finish()
