from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from hydra import compose, initialize
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse

from src.main_modules import (
    create_depthsplat_session,
    load_pretrained_weights,
    run_inference,
)


DEFAULT_OVERRIDES: list[str] = [
    "mode=test",
    "+experiment=dl3dv",
    "dataset/view_sampler=evaluation",
    "checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
    "dataset.roots=[/home/isarlab/PycharmProjects/incremental_splat/temp/output]",
    "dataset.image_shape=[540, 960]",
    "dataset.ori_image_shape=[540, 960]",
    "dataset.view_sampler.index_path=/home/isarlab/PycharmProjects/incremental_splat/temp/dl3dv.json",
    "model.encoder.num_scales=2",
    "model.encoder.upsample_factor=4",
    "model.encoder.lowest_feature_resolution=8",
    "model.encoder.monodepth_vit_type=vitb",
    "dataset.view_sampler.num_context_views=6",
    "test.stablize_camera=false",
    "test.compute_scores=false",
    "test.save_image=true",
    "test.compute_image=true",
    "test.save_depth=false",
    "test.process_pc=false",
    "output_dir=/home/isarlab/PycharmProjects/incremental_splat/temp/output/data",
]


def _coerce_overrides(overrides_param: Any) -> list[str]:
    if overrides_param is None:
        return list(DEFAULT_OVERRIDES)
    if isinstance(overrides_param, str):
        return [overrides_param]
    return [str(item) for item in overrides_param]


class DepthSplatPipeline:
    def __init__(self, overrides: Sequence[str]):
        overrides = list(overrides)

        t0 = time.time()
        with initialize(version_base=None, config_path="../config"):
            self.cfg = compose(config_name="main", overrides=overrides)
        t1 = time.time()

        self.session = create_depthsplat_session(self.cfg)
        t2 = time.time()

        load_pretrained_weights(self.session, stage="test")
        t3 = time.time()

        self.base_timings = {
            "config": t1 - t0,
            "session": t2 - t1,
            "weights": t3 - t2,
        }

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        inference_result = run_inference(self.session, load_weights=False) or {}
        t1 = time.time()

        occupancy_grid = inference_result.get("occupancy_grid")
        rendered_images = inference_result.get("rendered_images", [])

        timings = dict(self.base_timings)
        timings["inference"] = t1 - t0
        timings["total"] = sum(timings.values())

        return {
            "occupancy_grid": occupancy_grid,
            "rendered_images": rendered_images,
            "timings": timings,
        }


def _create_occupancy_msg(
    grid: np.ndarray,
    resolution: float,
    frame_id: str,
) -> OccupancyGrid:
    msg = OccupancyGrid()
    stamp = rospy.Time.now()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.info.resolution = resolution
    msg.info.width = int(grid.shape[1])
    msg.info.height = int(grid.shape[0])

    origin = Pose()
    origin.position = Point(
        x=-(msg.info.width * resolution) / 2.0,
        y=-(msg.info.height * resolution) / 2.0,
        z=0.0,
    )
    origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    msg.info.origin = origin

    flattened = (grid.astype(np.int16) * 100).clip(0, 100)
    msg.data = flattened.flatten().tolist()
    return msg


def _create_image_msg(image: np.ndarray, frame_id: str, seq: int) -> Image:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    msg = Image()
    msg.header = Header(stamp=rospy.Time.now(), frame_id=frame_id, seq=seq)
    msg.height, msg.width, channels = image.shape
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = msg.width * channels
    msg.data = image.tobytes()
    return msg


class DepthSplatROSNode:
    def __init__(self) -> None:
        overrides = _coerce_overrides(rospy.get_param("~hydra_overrides", DEFAULT_OVERRIDES))
        self.pipeline = DepthSplatPipeline(overrides)

        self.grid_resolution = float(rospy.get_param("~grid_resolution", 0.1))
        self.grid_frame = rospy.get_param("~grid_frame_id", "map")
        self.image_frame_prefix = rospy.get_param("~image_frame_prefix", "depthsplat_target")
        self.publish_grid = bool(rospy.get_param("~publish_grid", True))
        self.publish_images = bool(rospy.get_param("~publish_images", True))

        self.occupancy_pub = rospy.Publisher("~occupancy_grid", OccupancyGrid, queue_size=1, latch=True)
        self.images_pub = rospy.Publisher("~rendered_images", Image, queue_size=10)

        self.run_service = rospy.Service("~run", Trigger, self._handle_run_request)

        if rospy.get_param("~auto_run", True):
            rospy.loginfo("depthsplat_ros: auto-running inference")
            self._publish(self.pipeline.run())

    def _publish(self, result: dict[str, Any]) -> None:
        timings = result.get("timings", {})
        if timings:
            timings_str = ", ".join(f"{k}={v:.3f}s" for k, v in timings.items())
            rospy.loginfo(f"depthsplat_ros timings: {timings_str}")

        occupancy = result.get("occupancy_grid")
        if self.publish_grid and occupancy is not None:
            msg = _create_occupancy_msg(occupancy, self.grid_resolution, self.grid_frame)
            self.occupancy_pub.publish(msg)

        if self.publish_images:
            for seq, (target_id, image) in enumerate(result.get("rendered_images", []), start=1):
                frame_id = f"{self.image_frame_prefix}_{target_id}"
                msg = _create_image_msg(np.asarray(image), frame_id, seq)
                self.images_pub.publish(msg)

    def _handle_run_request(self, _: Trigger) -> TriggerResponse:
        try:
            rospy.loginfo("depthsplat_ros: received inference request")
            result = self.pipeline.run()
            self._publish(result)
            return TriggerResponse(success=True, message="DepthSplat inference completed")
        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"depthsplat_ros: inference failed - {exc}")
            return TriggerResponse(success=False, message=str(exc))


def run_depthsplat_pipeline(overrides: Sequence[str] | None = None) -> dict[str, Any]:
    pipeline = DepthSplatPipeline(overrides or DEFAULT_OVERRIDES)
    return pipeline.run()


def main() -> None:
    rospy.init_node("depthsplat_runner", anonymous=False)
    DepthSplatROSNode()
    rospy.loginfo("depthsplat_ros node ready")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
