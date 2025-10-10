from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from hydra import compose, initialize  # direttamente da Hydra

from src.main_modules import (
    create_depthsplat_session,
    load_pretrained_weights,
    run_inference,
)


def run_depthsplat_pipeline() -> dict[str, Any]:
    scale_factor = 1
    ori_image_shape = [540, 960]
    image_shape = [int(scale_factor * ori_image_shape[0]), int(scale_factor * ori_image_shape[1])]

    t_start = time.time()

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="main", overrides=["mode=test", "+experiment=dl3dv",
                                                     "dataset/view_sampler=evaluation",
                                                     "checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
                                                     "dataset.roots=[/home/isarlab/PycharmProjects/incremental_splat/temp/output]",
                                                     "dataset.image_shape=" + str(image_shape),
                                                     "dataset.ori_image_shape=" + str(ori_image_shape),
                                                     "dataset.view_sampler.index_path=/home/isarlab/PycharmProjects/incremental_splat/temp/dl3dv.json",
                                                     "model.encoder.num_scales=2",
                                                     "model.encoder.upsample_factor=4",
                                                     "model.encoder.lowest_feature_resolution=8",
                                                     "model.encoder.monodepth_vit_type=vitb",
                                                     "dataset.view_sampler.num_context_views=6", # %s" % str(len(context[0])+len(context[1])),
                                                     "test.stablize_camera=false",
                                                     "test.compute_scores=false",
                                                     "test.save_image=false",
                                                     "test.compute_image=true",
                                                     "output_dir=/home/isarlab/PycharmProjects/incremental_splat/temp/output/data",
                                                     "test.save_depth=false",
                                                     "test.process_pc=true",
                                                     "test.save_gaussian=true",
                                                     ])
    config_time = time.time()

    session = create_depthsplat_session(cfg)
    session_time = time.time()

    load_pretrained_weights(session, stage="test")
    weights_time = time.time()

    inference_result = run_inference(session, load_weights=False) or {}
    inference_time = time.time()

    occupancy_grid = inference_result.get("occupancy_grid")
    rendered_images = inference_result.get("rendered_images", [])

    return {
        "occupancy_grid": occupancy_grid,
        "rendered_images": rendered_images,
        "output_dir": session.output_dir,
        "timings": {
            "config": config_time - t_start,
            "session": session_time - config_time,
            "weights": weights_time - session_time,
            "inference": inference_time - weights_time,
            "total": inference_time - t_start,
        },
    }


if __name__ == "__main__":
    results = run_depthsplat_pipeline()
    occupancy = results["occupancy_grid"]
    grid_shape = None if occupancy is None else occupancy.shape
    rendered_images = results["rendered_images"]
    first_rendered_shape = rendered_images[0][1].shape if rendered_images else None
    print(f"Loaded occupancy grid shape: {grid_shape}")
    print(f"Loaded rendered image shape: {first_rendered_shape}")
    print("Timings (s):")
    for label, value in results["timings"].items():
        print(f"  {label}: {value:.3f}")

    plot = False
    if plot:
        if rendered_images:
            num_images = len(rendered_images)
            cols = min(num_images, 4)
            rows = int(np.ceil(num_images / cols))
            fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if not isinstance(axes2, np.ndarray):
                axes_list = [axes2]
            else:
                axes_list = axes2.flatten()

            for ax, (idx, image) in zip(axes_list, rendered_images):
                ax.imshow(image)
                ax.set_title(f"Rendered {idx}")
                ax.axis("off")

            for ax in axes_list[len(rendered_images):]:
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        else:
            print("No rendered images available for plotting.")
