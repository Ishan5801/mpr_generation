# =========================================================
# GPU-ACCELERATED STRAIGHTENED MPR GENERATION
# FIXED CENTERLINE ORDERING + NIFTI SHAPE ISSUE
# =========================================================

import os
import re
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation

# =========================================================
# CONFIG
# =========================================================

CT_DIR = r"D:\AICOE- Ishan\Codes\data\ICAS\ct_volumes"

CENTERLINE_DIR = (
    r"D:\AICOE- Ishan\Codes"
    r"\centerline_extraction_and_reconstruction"
    r"\outputs\run_12\centerlines\ICAS"
)

RADIUS_DIR = (
    r"D:\AICOE- Ishan\Codes"
    r"\centerline_extraction_and_reconstruction"
    r"\radius\run_12\ICAS"
)

OUTPUT_ROOT = "outputs"

RESAMPLE_STEP_MM = 0.5

CROSS_SECTION_SIZE = 64

PLANE_SCALE = 4.0

MAX_SLICES = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {DEVICE}")

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# =========================================================
# CREATE RUN FOLDER
# =========================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

existing_runs = glob.glob(
    os.path.join(OUTPUT_ROOT, "run_*")
)

if len(existing_runs) == 0:

    run_id = 1

else:

    run_nums = [
        int(re.findall(r"run_(\d+)", r)[0])
        for r in existing_runs
    ]

    run_id = max(run_nums) + 1

RUN_DIR = os.path.join(
    OUTPUT_ROOT,
    f"run_{run_id}"
)

STRAIGHTENED_CT_DIR = os.path.join(
    RUN_DIR,
    "straightened_ct"
)

os.makedirs(
    STRAIGHTENED_CT_DIR,
    exist_ok=True
)

# =========================================================
# UTILITIES
# =========================================================

def normalize(v):

    norm = np.linalg.norm(v)

    if norm < 1e-8:
        return v

    return v / norm


def load_nifti(path):

    nii = nib.load(path)

    data = nii.get_fdata()

    return data, nii.affine, nii.header


def save_nifti(data, affine, path):

    nii = nib.Nifti1Image(
        data.astype(np.float32),
        affine
    )

    nib.save(nii, path)


# =========================================================
# EXTRACT CENTERLINE POINTS
# =========================================================

def extract_centerline_points(centerline_volume):

    coords = np.argwhere(
        centerline_volume > 0
    )

    # ZYX -> XYZ
    coords = coords[:, [2, 1, 0]]

    if len(coords) == 0:
        return coords

    # =====================================================
    # GREEDY NEAREST-NEIGHBOR ORDERING
    # =====================================================

    ordered = [coords[0]]

    remaining = coords[1:].tolist()

    while len(remaining) > 0:

        last = ordered[-1]

        remaining_array = np.array(
            remaining
        )

        distances = np.linalg.norm(
            remaining_array - last,
            axis=1
        )

        idx = np.argmin(distances)

        ordered.append(
            remaining.pop(idx)
        )

    coords = np.array(ordered)

    return coords


# =========================================================
# SMOOTH + RESAMPLE CENTERLINE
# =========================================================

def smooth_resample_centerline(
    points,
    step=0.5
):

    points = np.array(points)

    if len(points) < 5:
        return points

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    tck, u = splprep(
        [x, y, z],
        s=5
    )

    unew = np.linspace(
        0,
        1,
        int(len(points) * 3)
    )

    out = splev(unew, tck)

    smooth_points = np.vstack(out).T

    distances = np.sqrt(
        np.sum(
            np.diff(
                smooth_points,
                axis=0
            ) ** 2,
            axis=1
        )
    )

    cumulative = np.insert(
        np.cumsum(distances),
        0,
        0
    )

    total_length = cumulative[-1]

    print(
        f"Estimated artery length: "
        f"{total_length:.2f} voxels"
    )

    # =====================================================
    # PREVENT EXPLOSION
    # =====================================================

    num_samples = min(
        int(total_length / step),
        MAX_SLICES
    )

    print(
        f"Resampled slices: "
        f"{num_samples}"
    )

    sample_d = np.linspace(
        0,
        total_length,
        num_samples
    )

    x_new = np.interp(
        sample_d,
        cumulative,
        smooth_points[:, 0]
    )

    y_new = np.interp(
        sample_d,
        cumulative,
        smooth_points[:, 1]
    )

    z_new = np.interp(
        sample_d,
        cumulative,
        smooth_points[:, 2]
    )

    resampled = np.vstack(
        [x_new, y_new, z_new]
    ).T

    return resampled


# =========================================================
# PARALLEL TRANSPORT FRAMES
# =========================================================

def compute_frames(points):

    tangents = []
    normals = []
    binormals = []

    up = np.array(
        [0, 0, 1],
        dtype=np.float32
    )

    prev_normal = None

    for i in range(len(points)):

        if i == len(points) - 1:

            tangent = (
                points[i]
                - points[i - 1]
            )

        else:

            tangent = (
                points[i + 1]
                - points[i]
            )

        tangent = normalize(tangent)

        tangents.append(tangent)

        if prev_normal is None:

            normal = np.cross(
                tangent,
                up
            )

            if np.linalg.norm(normal) < 1e-5:

                up = np.array([0, 1, 0])

                normal = np.cross(
                    tangent,
                    up
                )

            normal = normalize(normal)

        else:

            v = np.cross(
                prev_tangent,
                tangent
            )

            if np.linalg.norm(v) < 1e-5:

                normal = prev_normal

            else:

                v = normalize(v)

                angle = np.arccos(
                    np.clip(
                        np.dot(
                            prev_tangent,
                            tangent
                        ),
                        -1,
                        1
                    )
                )

                rot = Rotation.from_rotvec(
                    v * angle
                )

                normal = rot.apply(
                    prev_normal
                )

                normal = normalize(normal)

        binormal = np.cross(
            tangent,
            normal
        )

        binormal = normalize(binormal)

        normals.append(normal)
        binormals.append(binormal)

        prev_normal = normal
        prev_tangent = tangent

    return (
        np.array(tangents),
        np.array(normals),
        np.array(binormals)
    )


# =========================================================
# GPU STRAIGHTENED MPR
# =========================================================

def create_straightened_mpr_gpu(
    ct,
    centerline_points,
    radius_map
):

    centerline_points = (
        smooth_resample_centerline(
            centerline_points,
            step=RESAMPLE_STEP_MM
        )
    )

    tangents, normals, binormals = (
        compute_frames(
            centerline_points
        )
    )

    D, H, W = ct.shape

    print(
        f"\nFinal slice count: "
        f"{len(centerline_points)}"
    )

    # =====================================================
    # CT TO GPU
    # =====================================================

    ct_tensor = torch.tensor(
        ct,
        dtype=torch.float32,
        device=DEVICE
    )

    ct_tensor = ct_tensor.unsqueeze(0)
    ct_tensor = ct_tensor.unsqueeze(0)

    outputs = []

    print("\nRunning GPU interpolation...")

    for i, p in enumerate(centerline_points):

        if i % 100 == 0:

            print(
                f"Slice "
                f"{i}/{len(centerline_points)}"
            )

        x, y, z = p

        radius = 5

        try:

            radius = radius_map[
                int(z),
                int(y),
                int(x)
            ]

        except:
            pass

        plane_half = max(
            radius * PLANE_SCALE,
            8
        )

        coords = np.linspace(
            -plane_half,
            plane_half,
            CROSS_SECTION_SIZE
        )

        uu, vv = np.meshgrid(
            coords,
            coords
        )

        normal = normals[i]
        binormal = binormals[i]

        sample_points = (
            p[None, None, :]
            + uu[..., None]
            * normal[None, None, :]
            + vv[..., None]
            * binormal[None, None, :]
        )

        sx = sample_points[..., 0]
        sy = sample_points[..., 1]
        sz = sample_points[..., 2]

        # =================================================
        # NORMALIZE TO [-1, 1]
        # =================================================

        gx = (sx / (W - 1)) * 2 - 1
        gy = (sy / (H - 1)) * 2 - 1
        gz = (sz / (D - 1)) * 2 - 1

        grid = np.stack(
            [gx, gy, gz],
            axis=-1
        )

        grid_tensor = torch.tensor(
            grid,
            dtype=torch.float32,
            device=DEVICE
        )

        grid_tensor = (
            grid_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sampled = F.grid_sample(
            ct_tensor,
            grid_tensor,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        slice_img = sampled[
            0,
            0,
            0
        ]

        outputs.append(
            slice_img.cpu()
        )

    straightened = torch.stack(
        outputs
    )

    straightened = (
        straightened.numpy()
    )

    return straightened


# =========================================================
# PROCESS ALL CASES
# =========================================================

ct_files = sorted(
    glob.glob(
        os.path.join(
            CT_DIR,
            "*.nii*"
        )
    )
)

for ct_path in ct_files:

    base = os.path.basename(ct_path)

    # Example:
    # Img_003_0000.nii.gz
    #
    # -> Img_003

    match = re.match(
        r"(Img_\d+)",
        base
    )

    if match is None:

        print(
            f"Skipping invalid file: "
            f"{base}"
        )

        continue

    name = match.group(1)

    print("\n=================================")
    print(f"Processing {name}")
    print("=================================")

    centerline_path = os.path.join(
        CENTERLINE_DIR,
        f"{name}_centerlines_all.nii.gz"
    )

    radius_path = os.path.join(
        RADIUS_DIR,
        f"{name}_radius_map.nii.gz"
    )

    if not os.path.exists(centerline_path):

        print("Centerline not found")

        continue

    # =====================================================
    # LOAD DATA
    # =====================================================

    ct, affine, header = load_nifti(
        ct_path
    )

    centerline_vol, _, _ = load_nifti(
        centerline_path
    )

    if os.path.exists(radius_path):

        radius_map, _, _ = load_nifti(
            radius_path
        )

    else:

        radius_map = np.zeros_like(ct)

    centerline_points = (
        extract_centerline_points(
            centerline_vol
        )
    )

    print(
        f"Raw centerline points: "
        f"{len(centerline_points)}"
    )

    if len(centerline_points) < 5:

        print("Too few points")

        continue

    # =====================================================
    # GENERATE MPR
    # =====================================================

    straightened = (
        create_straightened_mpr_gpu(
            ct,
            centerline_points,
            radius_map
        )
    )

    print(
        f"\nOutput shape: "
        f"{straightened.shape}"
    )

    # =====================================================
    # SAVE
    # =====================================================

    output_path = os.path.join(
        STRAIGHTENED_CT_DIR,
        f"{name}_straightened_ct.nii.gz"
    )

    save_nifti(
        straightened,
        np.eye(4),
        output_path
    )

    print("\nSaved:")
    print(output_path)

print("\nDONE")