"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import datetime
import numpy as np
import torch as th
import torch.distributed as dist
import sys

from improved_diffusion_for_SMLM.improved_diffusion import dist_util, logger
from improved_diffusion_for_SMLM.improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main_args():
    args = create_argparser().parse_args()
    args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    main(**args_dict)
def main(logdir='',model_path='', image_size=64, num_channels = 64, num_res_blocks = 1, num_heads = 4, num_heads_upsample = -1,
         attention_resolutions = '16,8', dropout = 0.0, learn_sigma = False, sigma_small = False, class_cond = False,
         diffusion_steps = 1000, noise_schedule = 'cosine', timestep_respacing = '', use_kl = False,
         predict_xstart = False, rescale_timesteps = True, rescale_learned_sigmas = True, use_checkpoint = False,
         use_scale_shift_norm = True, batch_size = 10, num_samples = 10, clip_denoised = True, use_ddim = False):

    os.environ["OPENAI_LOGDIR"] = logdir + datetime.datetime.now().strftime(
        "model_sample-%Y-%m-%d-%H-%M-%S-%f")

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        class_cond=class_cond,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        #**args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    print("diffusion steps:", diffusion.num_timesteps)
    print("model var type:", diffusion.model_var_type)
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * batch_size < num_samples:
        model_kwargs = {}
        if class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (batch_size, 3, image_size, image_size),
            clip_denoised= clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    if class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=10,
        use_ddim=False,
        model_path='',
        logdir='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main_args()
