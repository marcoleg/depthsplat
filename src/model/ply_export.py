from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


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

    # import matplotlib.pyplot as plt
    # plt.imshow(grid, cmap="gray_r", origin="lower")
    # plt.title("Occupancy Grid")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.scatter(y_points, x_points)
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
):

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

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
    # if save_pc:
    #     plydata.write(path)
    if process_pc:
        v = plydata['vertex'].data
        arr = np.array(v)
        # print('v original shape:', v.shape, '\nx: {} ; {}\ny: {} ; {}\nz: {} {}'.format(arr['x'].min(),
        #                                                                               arr['x'].max(),
        #                                                                               arr['y'].min(),
        #                                                                               arr['y'].max(),
        #                                                                               arr['z'].min(),
        #                                                                               arr['z'].max()))
        mask = (
                (arr['x'] <= 20.0) &
                (arr['x'] >= -20.0) &
                (arr['y'] <= 5.0) &
                (arr['y'] >= -5.0) &
                (arr['z'] <= 25.0) &
                (arr['z'] >= -25.0)
        )
        close_points = arr[mask]
        # print('close points:', close_points.shape)
        opacity = np.array(close_points['opacity'], dtype=np.float32)
        mask = opacity > -1.5
        filtered = close_points[mask]
        plydata['vertex'].data = filtered
        if save_pc:
            plydata.write(str(path).split('.ply')[0] + '_FILTERED.ply')

        np.save(str(path).split('.ply')[0] + '_grid.npy', pointcloud_to_occupancy_grid(points=filtered))

def save_gaussian_ply(gaussians, visualization_dump, example, save_path, save_pc, process_pc):

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

    export_ply(
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


