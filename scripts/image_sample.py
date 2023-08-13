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
sys.path.append('/data/GAN_project/scripts')

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main(model_path, output_path, diffusion_steps, patch_size, images_num):
    os.environ["OPENAI_LOGDIR"] = output_path
    args = create_argparser(model_path,images_num).parse_args()

    dist_util.setup_dist()
    logger.configure()
    args.image_size = patch_size
    args.diffusion_steps = diffusion_steps
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    print("diffusion steps:", diffusion.num_timesteps)
    print("model var type:", diffusion.model_var_type)
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser(model_path, images_num):
    defaults = dict(
        clip_denoised=True,
        num_samples=images_num,
        batch_size=10,
        use_ddim=False,
        model_path=model_path,
        # image_size=64,
        # num_channels=64,
        # num_res_blocks=1,
        # noise_schedule="linear",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    model_path = "/data/GAN_project/diffusion_tries/mitochondria/tav/openai-2023-04-25-18-02-13-010307/ema_0.9999_080000.pt"
    #main(model_path, output_path, diffusion_steps=2000, patch_size=256, images_num=10)
    # with np.load('/data/GAN_project/diffusion_tries/samples/openai-2023-04-26-15-04-29-012512/samples_10x256x256x3.npz') as data:
    #    lst = data.files
    #    for item in data[lst[0]]:
    #        plt.imshow(item)
    #        plt.show()
    patch_size = 256
    diffusion_steps = 2000 #2000
    model_path = '/data/GAN_project/diffusion_tries/mitochondria/shareloc/openai-2023-06-30-08-38-22-542186/ema_0.9999_148000.pt'
    model_path = '/data/GAN_project/diffusion_tries/mitochondria/shareloc/7_imgsopenai-2023-07-06-19-38-06-665622/ema_0.9999_080000.pt'

    model_path ='/data/GAN_project/diffusion_tries/mitochondria/shareloc/7_imgs_not_saturated_openai-2023-07-15-09-25-31-059997/ema_0.9999_062000.pt'
    for i in range(1, 10):
        #output_path = "/data/GAN_project/diffusion_tries/samples/mitochondria/1106/" + datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
        output_path = '/data/GAN_project/diffusion_tries/samples/mitochondria/shareloc/7imgs_not_saturated1507' + datetime.datetime.now().strftime(
            "openai-%Y-%m-%d-%H-%M-%S-%f")
        images_num = 300
        main(model_path, output_path, diffusion_steps, patch_size, images_num)
    # output_path = '/data/GAN_project/diffusion_tries/samples/mitochondria/shareloc/3006'
    #output_path = '/data/GAN_project/diffusion_tries/samples/mitochondria/shareloc/0707_7imgs'
    output_path = '/data/GAN_project/diffusion_tries/samples/mitochondria/shareloc/0707_7imgs_not_saturated1507_2'
    images_num = 10
    #main(model_path, output_path, diffusion_steps,patch_size, images_num)
