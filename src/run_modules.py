from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import argparse
import threading
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import shared_memory

from hydra import compose, initialize  # direttamente da Hydra

from src.main_modules import (
    create_depthsplat_session,
    load_pretrained_weights,
    run_inference,
)


def _to_shared(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr  # copy
    return {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
    }, shm  # return meta + handle to close/unlink later

def _prep_payload_shared(occupancy_grid, rendered_images):
    """
    rendered_images è una lista di (idx, image_ndarray).
    Impacchettiamo le immagini in un unico array (N,H,W,C) e passiamo gli indici a parte.
    """
    metas = {}
    shms = []

    # Occupancy grid (opzionale)
    if occupancy_grid is not None:
        if not isinstance(occupancy_grid, np.ndarray):
            occupancy_grid = np.asarray(occupancy_grid)
        og_meta, og_shm = _to_shared(occupancy_grid)
        metas["occupancy_grid"] = og_meta
        shms.append(og_shm)
    else:
        metas["occupancy_grid"] = None

    # Rendered images (lista)
    indices = []
    imgs = []
    for idx, img in (rendered_images or []):
        indices.append(int(idx))
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        imgs.append(img)

    if imgs:
        # Verifica omogeneità e stack
        first_shape = imgs[0].shape
        first_dtype = imgs[0].dtype
        if not all(im.shape == first_shape for im in imgs):
            raise ValueError("Tutte le rendered_images devono avere la stessa shape per lo stack in shared memory.")
        if not all(im.dtype == first_dtype for im in imgs):
            imgs = [im.astype(first_dtype) for im in imgs]
        img_block = np.stack(imgs, axis=0)  # (N,H,W[,C])
        img_meta, img_shm = _to_shared(img_block)
        metas["rendered_images"] = img_meta
        metas["rendered_indices"] = indices
        shms.append(img_shm)
    else:
        metas["rendered_images"] = None
        metas["rendered_indices"] = []

    return metas, shms


class CommandServer:
    """Simple TCP server that logs JSON commands received from ROS subscriber."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.host = host
        self.port = port
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.output_dict = {'flag': False, 'seq_name': None, 'context': None, 'target': None}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(1.0)
            print(f"[CommandServer] Listening on {self.host}:{self.port}")
            while not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def _handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        print(f"[CommandServer] Client connected: {addr}")
        with conn:
            buffer = ""
            while not self._stop_event.is_set():
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[CommandServer] Bad packet from {addr}: {exc} :: {line!r}")
                        continue
                    flag = payload.get("trigger")
                    seq_name = payload.get("seq_name")
                    context = payload.get("context")
                    target = payload.get("target")
                    print(f"[CommandServer] trigger={flag}, seq_name={seq_name}, context={context}, target={target}")
                    self.output_dict = {'flag': flag, 'seq_name': seq_name, 'context': context, 'target': target}
        print(f"[CommandServer] Client disconnected: {addr}")

def launch_consumer_different_interp(python_exe: str, consumer_script: str, meta: dict) -> int:
    """
    Avvia il consumer in un interprete diverso, passando il meta JSON via argv.
    """
    args = [python_exe, consumer_script, json.dumps(meta)]
    proc = subprocess.run(args, text=True)
    return proc.returncode

def run_depthsplat_pipeline() -> object:
    scale_factor = 1
    ori_image_shape = [480, 640]  # airsim
    image_shape = [int(scale_factor * ori_image_shape[0]), int(scale_factor * ori_image_shape[1])]

    t_start = time.time()

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="main", overrides=["mode=test", "+experiment=dl3dv",
                                                     "dataset/view_sampler=evaluation",
                                                     "checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
                                                     "dataset.roots=[/root/incremental_splat/ros_ws/src/incremental_splat/src/temp/output]",
                                                     "dataset.image_shape=" + str(image_shape),
                                                     "data_loader.test.num_workers=0",
                                                     "dataset.ori_image_shape=" + str(ori_image_shape),
                                                     "dataset.view_sampler.index_path=/root/incremental_splat/ros_ws/src/incremental_splat/src/temp/dl3dv.json",
                                                     "model.encoder.num_scales=2",
                                                     "model.encoder.upsample_factor=4",
                                                     "model.encoder.lowest_feature_resolution=8",
                                                     "model.encoder.monodepth_vit_type=vitb",
                                                     "dataset.view_sampler.num_context_views=6", # %s" % str(len(context[0])+len(context[1])),
                                                     "test.stablize_camera=false",
                                                     "test.compute_scores=false",
                                                     "test.save_image=true",
                                                     "test.compute_image=true",
                                                     "output_dir=/root/incremental_splat/ros_ws/src/incremental_splat/src/temp/output/data",
                                                     "test.save_depth=true",
                                                     "test.process_pc=true",
                                                     "test.save_gaussian=true",
                                                     ])
    config_time = time.time()

    session = create_depthsplat_session(cfg)
    session_time = time.time()

    load_pretrained_weights(session, stage="test")
    weights_time = time.time()
    return session, config_time, session_time, weights_time, t_start

def prepare_ds_inputs():
    command = [
        "python",
        "/root/depthsplat/src/scripts/convert_dl3dv_test.py",
        "--input_dir", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input'),
        "--output_dir", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output'),
    ]
    subprocess.run(command, check=True)

    command = [
        "python",
        "/root/depthsplat/src/scripts/generate_dl3dv_index.py",
        "--dataset_path", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output'),
    ]
    subprocess.run(command, check=True)

def run_difix(seq_name, context, target, n, context_dir_name=None):
    print('\n############# Running Difix')
    if context_dir_name is None:
        str_context = str(context)
        context_dir_name = str_context.replace(', ', '-').replace('[', '').replace(']', '')

    ref_frame_name = f'frame_{context[0]:0>5}.png'
    frame_name = f'frame_new_{n}.png'

    assert len(target) == 1, "Target should contain exactly one frame index."  # TODO: generalizza per len(target) > 1
    shutil.move(os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'images', seq_name, context_dir_name, f'frame_{target[0]+1:0>5}.png'),
                os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'images', seq_name, context_dir_name, frame_name))

    python_exec = "/root/miniconda3/envs/depthsplat/bin/python"

    command = [
        python_exec,
        "/root/Difix3D/src/inference_difix.py",
        "--input_image", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'images', seq_name, context_dir_name, frame_name),
        "--height", "640",  # 360 -> 320
        "--width", "480",   # 640 -> 576
        "--prompt", "",
        "--model_path", "/root/Difix3D/outputs/difix/train/checkpoints/model_v3.pkl",
        "--output_dir", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input', seq_name, 'gaussian_splat', 'images_4'),
        "--ref_image", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input', seq_name, 'gaussian_splat', 'images_4', ref_frame_name)
    ]

    subprocess.run(command, check=True)

    _update_transforms_json(seq_name, n)

def _update_transforms_json(seq_name, n):
    trans_path = os.path.join("/root/incremental_splat/ros_ws/src/incremental_splat/src/temp", "input", seq_name, "gaussian_splat", "transforms.json")
    transforms = json.load(open(trans_path, 'r'))
    new_tf = [{
        "file_path": f"images/frame_new_{n}.png",
        "transform_matrix": [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [-0.0, -0.0, -1.0, -0.0]]  # TODO: aggiorna con posa corretta
    }]
    transforms['frames'] += new_tf
    json.dump(transforms, open(trans_path, 'w'))  # type: ignore[arg-type]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, help="Times that we take (last+K)-th frame", default=0)
    args = parser.parse_args()

    session, config_time, session_time, weights_time, t_start = run_depthsplat_pipeline()
    inference_time = time.time()
    cmd_host = os.environ.get("DEPTHSPLAT_CMD_HOST", "127.0.0.1")
    cmd_port = int(os.environ.get("DEPTHSPLAT_CMD_PORT", "5555"))
    command_server = CommandServer(host=cmd_host, port=cmd_port)
    command_server.start()
    print('Our output_dict:', command_server.output_dict)
    print(f"[CommandServer] Ready on {cmd_host}:{cmd_port}")

    try:
        print("[CommandServer] Listening for incoming triggers (Ctrl+C to stop).")
        while True:
            # print("[", time.time(), "]Current flag status:", command_server.output_dict['flag'])
            if command_server.output_dict['flag']:
                print("Received trigger:", command_server.output_dict)
                prepare_ds_inputs()
                inference_result = run_inference(session, load_weights=False) or {}

                if args.N > 0:
                    for n in range (args.N):
                        run_difix(seq_name=command_server.output_dict.get('seq_name'), 
                                  context=command_server.output_dict.get('context'),
                                  target=command_server.output_dict.get('target'),
                                  n=n)
                        inference_result = run_inference(session, load_weights=False) or {}
                    assert False, "now u have to send occupancy grid and rendered images via shared memory"

                occupancy_grid = inference_result.get("occupancy_grid")
                rendered_images = inference_result.get("rendered_images", [])
                results = {
                    "occupancy_grid": occupancy_grid,
                    "rendered_images": rendered_images,
                    "output_dir": session.output_dir,
                    "timings": {
                        "config": config_time - t_start,
                        "session": session_time - config_time,
                        "weights": weights_time - session_time,
                        "inference": inference_time - weights_time,
                        "total": inference_time - t_start}
                }

                occupancy = results["occupancy_grid"]
                grid_shape = None if occupancy is None else occupancy.shape
                rendered_images = results["rendered_images"]
                first_rendered_shape = rendered_images[0][1].shape if rendered_images else None
                print(f"Loaded occupancy grid shape: {grid_shape}")
                print(f"Loaded rendered image shape: {first_rendered_shape}")
                print("Timings (s):")
                for label, value in results["timings"].items():
                    print(f"  {label}: {value:.3f}")

                meta, shm_handles = _prep_payload_shared(occupancy, rendered_images)
                other_python = "/usr/bin/python3"
                consumer_script = "/root/incremental_splat/ros_ws/src/depthsplat_ros/src/consumer.py"

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

                try:
                    rc = launch_consumer_different_interp(other_python, consumer_script, meta)
                    print("Consumer return code:", rc)
                finally:
                    for shm in shm_handles:
                        try:
                            shm.close()
                        except Exception:
                            pass
                        try:
                            shm.unlink()
                        except Exception:
                            pass
                
                command_server.output_dict['flag'] = False  # reset flag after handling

    except KeyboardInterrupt:
        print("Shutting down due to KeyboardInterrupt.")
    finally:
        command_server.stop()
