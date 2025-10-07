from hydra import initialize, compose   # direttamente da Hydra
from src.main_modules import create_depthsplat_session, load_pretrained_weights, run_inference
import time

scale_factor = 1
ori_image_shape = [540,960]
image_shape = [int(scale_factor * ori_image_shape[0]), int(scale_factor * ori_image_shape[1])]

# Inizializza Hydra e crea la configurazione (qui mode=test + esperimento dl3dv)
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
                                                 "test.save_image=true",
                                                 "output_dir=/home/isarlab/PycharmProjects/incremental_splat/temp/output/data",
                                                 "test.save_depth=true"
                                                 ])
t0 = time.time()
# 1) Crea la sessione (costruisce modello, trainer, datamodule… ma non lancia ancora nulla)
session = create_depthsplat_session(cfg)
print(f"Session creation time: {time.time() - t0:.2f} seconds")

# 2) Carica i checkpoint/pretrained una volta sola
load_pretrained_weights(session, stage="test")
print(f"Checkpoint loading time: {time.time() - t0:.2f} seconds")

# 3) Quando vuoi eseguire l’inferenza
run_inference(session)
print(f"Total time: {time.time() - t0:.2f} seconds")
