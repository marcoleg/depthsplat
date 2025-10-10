from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor
import open3d as o3d
import time, warnings
from ..global_cfg import get_cfg


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def pointcloud_to_occupancy_grid(points, grid_size=0.1, width=8, height=8, z_threshold=-0.15, z_max=0.5):
    """
    Convert 3D point cloud to 2D occupancy grid.

    Args:
        points (np.ndarray): Nx3 array of 3D points (x,y,z).
        grid_size (float): resolution of the grid in meters.
        width (float): total width of the map in meters.
        height (float): total height of the map in meters.
        z_threshold (float): minimum z value to consider as obstacle.

    Returns:
        np.ndarray: 2D occupancy grid (0 = free, 1 = occupied).
    """
    t0 = time.time()
    # Grid dimensions
    cols = int(width / grid_size)
    rows = int(height / grid_size)
    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Shift points so that map is centered
    x_points = points["z"] + width / 2.0
    y_points = -points["x"] + height / 2.0
    z_points = -points["y"]

    # Filter points within map bounds and above threshold
    mask = (
        (x_points >= 0) & (x_points < width) &
        (y_points >= 0) & (y_points < height) &
        (z_points > z_threshold) & (z_points < z_max)
    )
    x_points = x_points[mask]
    y_points = y_points[mask]

    # Convert metric coordinates to grid indices
    col_idx = (x_points / grid_size).astype(int)
    row_idx = (y_points / grid_size).astype(int)

    # Mark cells as occupied
    grid[row_idx, col_idx] = 1
    print(f"[pointcloud_to_occupancy_grid] Generated grid with {np.sum(grid)} occupied cells"
          f" | time: {round(time.time() - t0, 3)}s")
    # import matplotlib.pyplot as plt
    # plt.imshow(grid, cmap="gray_r", origin="lower")
    # plt.title("Occupancy Grid")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # # plt.scatter(y_points, x_points)
    # plt.savefig('/root/incremental_splat/ros_ws/src/incremental_splat/src/occupancy_grid.png')

    return grid

def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    save_pc: bool,
    process_pc: bool,
) -> np.ndarray | None:

    t0 = time.time()
    try:
        cfg = get_cfg()
        out_cfg = getattr(cfg, "outlier", None)
    except Exception as e:
        warnings.warn(f"Errore nel caricamento config: {e}")
        out_cfg = None


    if out_cfg is not None:
        outlier_method = getattr(out_cfg, "outlier_method")
        radius = getattr(out_cfg, "radius")
        min_neighbors = getattr(out_cfg, "min_neighbors")
        nb_neighbors = getattr(out_cfg, "nb_neighbors")
        std_ratio = getattr(out_cfg, "std_ratio")


    def _mask_radius_xyz(xyz: np.ndarray, radius, min_neighbors) -> np.ndarray:
        """True for inliers >= min_neighbors within radius"""
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        _, idx_inliers = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        mask = np.zeros(len(xyz), dtype=bool)
        mask[np.asarray(idx_inliers, dtype=int)] = True
        return mask

    def _mask_stat_xyz(xyz: np.ndarray, nb_neighbors, std_ratio) -> np.ndarray:
        """True for inliers with the closest nb_neighbors"""
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        _, idx_inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        mask = np.zeros(len(xyz), dtype=bool)
        mask[np.asarray(idx_inliers, dtype=int)] = True
        return mask

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations # type: ignore
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1) # type: ignore

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        torch.logit(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    plydata = PlyData([PlyElement.describe(elements, "vertex")])

    occupancy_grid: np.ndarray | None = None
    print(f"[export_ply] Exporting {len(elements)} Gaussians | time: {round(time.time() - t0, 3)}s")
    if process_pc:
        t0 = time.time()
        v = plydata['vertex'].data
        arr = np.array(v)
        mask = (
                (arr['x'] <= 20.0) &
                (arr['x'] >= -20.0) &
                (arr['y'] <= 5.0) &
                (arr['y'] >= -5.0) &
                (arr['z'] <= 25.0) &
                (arr['z'] >= -25.0))
        close_points = arr[mask]
        opacity = np.array(close_points['opacity'], dtype=np.float32)
        mask = opacity > -1.5
        filtered = close_points[mask]

        if outlier_method in ("radi", "stat"):
            xyz = np.vstack([filtered['x'], filtered['y'], filtered['z']]).T.astype(np.float64)
            if outlier_method == "radi":
                mask_o3d = _mask_radius_xyz(xyz, radius=radius, min_neighbors=min_neighbors)
            else:
                mask_o3d = _mask_stat_xyz(xyz, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            filtered = filtered[mask_o3d]

        plydata['vertex'].data = filtered

        print(f"[export_ply] kept {len(filtered)} / {len(arr)} points ({len(filtered) / len(arr):.1%}) | time: "
              f"{round(time.time() - t0, 3)}s | method: {outlier_method}")
        if save_pc:
            plydata.write(str(path).split('.ply')[0] + '_FILTERED.ply')

        occupancy_grid = pointcloud_to_occupancy_grid(points=filtered)
        # np.save(str(path).split('.ply')[0] + '_grid.npy', occupancy_grid)

    return occupancy_grid

def save_gaussian_ply(gaussians, visualization_dump, example, save_path, save_pc, process_pc):
    t0 = time.time()

    v, _, h, w = example["context"]["image"].shape[1:]

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
    )

    # Create a mask to filter the Gaussians. throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 8
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    def trim(element):
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
        )
        return element[mask][None]

    # convert the rotations from camera space to world space as required
    cam_rotations = trim(visualization_dump["rotations"])[0]
    c2w_mat = repeat(
        example["context"]["extrinsics"][0, :, :3, :3],
        "v a b -> h w spp v a b",
        h=h,
        w=w,
        spp=1,
    )
    c2w_mat = c2w_mat[mask]  # apply trim

    cam_rotations_np = R.from_quat(
        cam_rotations.detach().cpu().numpy()
    ).as_matrix()
    world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
    world_rotations = R.from_matrix(world_mat).as_quat()
    world_rotations = torch.from_numpy(world_rotations).to(
        visualization_dump["scales"]
    )
    print(f"[save_gaussian_ply] Preprocess data | time: {round(time.time() - t0, 3)}s")
    t0 = time.time()
    eply = export_ply(
        example["context"]["extrinsics"][0, 0],
        trim(gaussians.means)[0],
        trim(visualization_dump["scales"])[0],
        world_rotations,
        trim(gaussians.harmonics)[0],
        trim(gaussians.opacities)[0],
        save_path,
        save_pc,
        process_pc,
    )
    print(f"[save_gaussian_ply] Export .PLY | time: {round(time.time() - t0, 3)}s")
    return eply

