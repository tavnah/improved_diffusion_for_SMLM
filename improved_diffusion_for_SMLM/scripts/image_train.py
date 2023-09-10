"""
Train a diffusion model on images.
"""

import argparse
import os
import numpy as np
import datetime
from improved_diffusion_for_SMLM.improved_diffusion import dist_util, logger
from improved_diffusion_for_SMLM.improved_diffusion.image_datasets import load_data
from improved_diffusion_for_SMLM.improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion_for_SMLM.improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion_for_SMLM.improved_diffusion.train_util import TrainLoop
import torch

def main_args():
    args = create_argparser().parse_args()
    args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    main(**args_dict, logdir=args.logdir, data_dir=args.data_dir)

def main(model_name = datetime.datetime.now().strftime("model_train-%Y-%m-%d-%H-%M-%S-%f"), logdir='',data_dir='',
         num_steps=-1, image_size=64, num_channels = 64, num_res_blocks = 1, num_heads = 4, num_heads_upsample = -1,
         attention_resolutions = '16,8', dropout = 0.0, learn_sigma = False, sigma_small = False, class_cond = False,
         diffusion_steps = 1000, noise_schedule = 'cosine', timestep_respacing = '', use_kl = False,
         predict_xstart = False, rescale_timesteps = True, rescale_learned_sigmas = True, use_checkpoint = False,
         use_scale_shift_norm = True, batch_size = 10, microbatch = -1, lr = 1e-4, ema_rate="0.9999",
         schedule_sampler = "uniform", log_interval=500, save_interval=2000, resume_checkpoint='', use_fp16=False,
         fp16_scale_growth = 1e-3, weight_decay=0.0, lr_anneal_steps=0):
    #args = create_argparser().parse_args()

    os.environ[
        "OPENAI_LOGDIR"] = logdir + model_name
        #datetime.datetime.now().strftime("model_train-%Y-%m-%d-%H-%M-%S-%f")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size = image_size,
        num_channels= num_channels,
        num_res_blocks= num_res_blocks,
        num_heads= num_heads,
        num_heads_upsample= num_heads_upsample,
        attention_resolutions= attention_resolutions,
        dropout= dropout,
        learn_sigma= learn_sigma,
        sigma_small= sigma_small,
        class_cond= class_cond,
        diffusion_steps= diffusion_steps,
        noise_schedule= noise_schedule,
        timestep_respacing= timestep_respacing,
        use_kl= use_kl,
        predict_xstart= predict_xstart,
        rescale_timesteps= rescale_timesteps,
        rescale_learned_sigmas= rescale_learned_sigmas,
        use_checkpoint= use_checkpoint,
        use_scale_shift_norm= use_scale_shift_norm,

    )
    logger.log(f"number of model parameters:{sum([np.prod(p.size()) for p in model.parameters()])}")
    logger.log(f"channel multiplier: {model.channel_mult}")
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
    )
    logger.log(f"data dir: {data_dir}")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
    ).run_loop(num_steps=num_steps)


def create_argparser():
    defaults = dict(data_dir='/data/GAN_project/mitochondria/shareloc/tiff_files/train/patches_256x256_ol0.25',
        model_name = datetime.datetime.now().strftime("model_train-%Y-%m-%d-%H-%M-%S-%f"),
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        save_interval=2000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        logdir = '',

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main_args()
