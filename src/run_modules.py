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
from scipy.spatial.transform import Rotation as R

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

def _prep_payload_shared(occupancy_grid):
    """
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

    return metas, shms


class CommandServer:
    """Simple TCP server that logs JSON commands received from ROS subscriber."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.host = host
        self.port = port
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.output_dict = {'flag': False, 'seq_name': None, 'context': None, 'target': None, 'last_pose': None}

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
                    last_pose = payload.get("last_pose")
                    print(f"[CommandServer] trigger={flag}, seq_name={seq_name}, context={context}, target={target}")
                    self.output_dict = {'flag': flag, 'seq_name': seq_name, 'context': context, 'target': target, 'last_pose': last_pose}
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
                                                     # "train.forward_depth_only=true",
                                                     # "checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth",
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
    cfg.dataset.target_names = ["new_0"]

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

def run_difix(seq_name, context, new_frame_name, context_dir_name=None):
    print('\n############# Running Difix')

    ref_frame_name = f'frame_{context[0]:0>5}.png'

    python_exec = "/root/miniconda3/envs/depthsplat/bin/python"

    command = [
        python_exec,
        "/root/Difix3D/src/inference_difix.py",
        "--input_image", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'images', seq_name, context_dir_name, new_frame_name),
        "--height", "480",  # 360 -> 320
        "--width", "640",   # 640 -> 576
        "--prompt", "",
        "--model_path", "/root/Difix3D/outputs/difix/train/checkpoints/model_v3.pkl",
        "--output_dir", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input', seq_name, 'gaussian_splat', 'images_4'),
        "--ref_image", os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input', seq_name, 'gaussian_splat', 'images_4', ref_frame_name)
    ]

    subprocess.run(command, check=True)

def _update_transforms_json(seq_name, new_frame_name, new_pose):
    trans_path = os.path.join("/root/incremental_splat/ros_ws/src/incremental_splat/src/temp", "input", seq_name, "gaussian_splat", "transforms.json")
    transforms = json.load(open(trans_path, 'r'))
    new_tf = [{
        "file_path": f"images/{new_frame_name}",
        "transform_matrix": new_pose
    }]
    transforms['frames'] += new_tf
    json.dump(transforms, open(trans_path, 'w'))  # type: ignore[arg-type]

def prepare_ds_newframes(session, command_server, T_rel, index, last_pose=None, context=None):

    session.cfg.dataset.target_names = ["new_%d" % index]

    if context:
        session.cfg.dataset.context_indices = context

    if last_pose is None:
        last_pose = command_server.output_dict.get('last_pose')

    new_pose = np.array(last_pose) @ T_rel  # ned reference system
    ned2opencv = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    target_pose = new_pose @ ned2opencv.T  # opencv reference system
    # target_pose = new_pose 
    session.cfg.dataset.target_poses = [target_pose.tolist()]
    _update_transforms_json(command_server.output_dict.get('seq_name'), f'frame_new_{index}.png', new_pose=new_pose.tolist())
    last_pose = new_pose
    str_context = str(command_server.output_dict.get('context'))
    context_dir_name = str_context.replace(', ', '-').replace('[', '').replace(']', '')
    shutil.copy(os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input', 
                            command_server.output_dict.get('seq_name'), 'gaussian_splat', 'images_4',
                            f'frame_{command_server.output_dict.get("context")[0]:0>5}.png'),
                        os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'input',
                            command_server.output_dict.get('seq_name'), 'gaussian_splat', 'images_4', f'frame_new_{index}.png'))

    if index > 0:
        new_dl3dv_json = {command_server.output_dict.get('seq_name'): {
            "context": command_server.output_dict.get('context'),
            "target": [0]
        }}
        dl3dv_json_path = os.path.join("/root/incremental_splat/ros_ws/src/incremental_splat/src/temp", "dl3dv.json")
        json.dump(new_dl3dv_json, open(dl3dv_json_path, 'w'))  # type: ignore[arg-type]

    return last_pose, context_dir_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, help="Times that we take (last+K)-th frame", default=0)
    parser.add_argument("--target_rpy", type=float, help="Required Roll, Yaw, Pitch", nargs="+", default=[0, 0, 0])
    parser.add_argument("--target_xyz", type=float, help="Required x (right), y (down), z (forward)", nargs="+", default=[0.0, 0, 0.0])
    args = parser.parse_args()

    # Compute absolute transformation matrix of new frame
    rot_mat = R.from_euler('xyz', args.target_rpy, degrees=True).as_matrix().tolist()
    T_rel = np.array([rot_mat[0]+[args.target_xyz[0]], rot_mat[1]+[args.target_xyz[1]], rot_mat[2]+[args.target_xyz[2]], [0.0, 0.0, 0.0, 1.0]])

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

            if command_server.output_dict['flag']:
                print(command_server.output_dict.get('context'))
                print("Received trigger:", command_server.output_dict['flag'], '\nSeq_name:', command_server.output_dict['seq_name'],)
                
                seq_name = command_server.output_dict.get('seq_name')
                context = command_server.output_dict.get('context')
                target = command_server.output_dict.get('target')

                last_pose, context_dir_name = prepare_ds_newframes(session, command_server, T_rel, index=0) 
                prepare_ds_inputs()

                inference_result = run_inference(session, load_weights=False) or {}
                shutil.move(os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'gaussians', seq_name + '_FILTERED.ply'),
                            os.path.join('/root/incremental_splat/ros_ws/src/incremental_splat/src/temp', 'output', 'data', 'gaussians', f'ORIG_{seq_name}_FILTERED.ply'))
                if args.N > 0:
                    for n in range (args.N):
                        run_difix(seq_name=seq_name, 
                                  context=context,
                                  new_frame_name=f'frame_new_{n}.png',
                                  context_dir_name=context_dir_name)
                        if n == args.N - 1:
                            session.cfg.test.save_gaussian = True
                        else:
                            session.cfg.test.save_gaussian = False
                        
                        context.append(target[0] + n)
                        last_pose, context_dir_name = prepare_ds_newframes(session, command_server, T_rel, index=n+1, last_pose=last_pose, context=context)
                        prepare_ds_inputs()
                        inference_result = run_inference(session, load_weights=False) or {}

                occupancy_grid = inference_result.get("occupancy_grid")
                results = {
                    "occupancy_grid": occupancy_grid,
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
                print(f"Loaded occupancy grid shape: {grid_shape}")
                print("Timings (s):")
                for label, value in results["timings"].items():
                    print(f"  {label}: {value:.3f}")

                meta, shm_handles = _prep_payload_shared(occupancy)
                other_python = "/usr/bin/python3"
                consumer_script = "/root/incremental_splat/ros_ws/src/depthsplat_ros/src/consumer.py"

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
