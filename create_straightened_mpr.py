# =========================================================
# GPU-ACCELERATED SEPARATE ARTERY STRAIGHTENED MPR
# =========================================================
#
# INPUTS:
#   1. CT Volume
#   2. Artery 1 Centerline
#   3. Artery 2 Centerline
#   4. Shared Radius Map
#
# CENTERLINE NAMES:
#   Img_001_artery_1_centerline.nii.gz
#   Img_001_artery_2_centerline.nii.gz
#
# OUTPUTS:
#   outputs/run_X/
#       artery_1/
#       artery_2/
#
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

CT_DIR = (
    r"D:\AICOE- Ishan\Codes\data_trail\ct"
)

CENTERLINE_DIR = (
    r"D:\AICOE- Ishan\Codes\data_trail\centerline"
)

RADIUS_DIR = (
    r"D:\AICOE- Ishan\Codes\data_trail\radius"
)

OUTPUT_ROOT = "outputs"

# =========================================================
# PARAMETERS
# =========================================================

RESAMPLE_STEP_MM = 1.0

CROSS_SECTION_SIZE = 48

PLANE_SCALE = 4.0

MAX_SLICES = 2500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {DEVICE}")

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# =========================================================
# CREATE OUTPUT RUN
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

ARTERY1_OUT_DIR = os.path.join(
    RUN_DIR,
    "artery_1"
)

ARTERY2_OUT_DIR = os.path.join(
    RUN_DIR,
    "artery_2"
)

os.makedirs(ARTERY1_OUT_DIR, exist_ok=True)
os.makedirs(ARTERY2_OUT_DIR, exist_ok=True)

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

    data = data.astype(np.float32)

    nii = nib.Nifti1Image(
        data,
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
    # GREEDY ORDERING
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
# SMOOTH + RESAMPLE
# =========================================================

def smooth_resample_centerline(
    points,
    step=1.0
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
        f"{total_length:.2f}"
    )

    num_samples = min(
        int(total_length / step),
        MAX_SLICES
    )

    print(
        f"Final slice count: "
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
# GPU MPR
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

        # =================================================
        # RADIUS
        # =================================================

        try:

            radius = radius_map[
                int(z),
                int(y),
                int(x)
            ]

            if radius <= 0:
                radius = 5

        except:

            radius = 5

        plane_half = max(
            radius * PLANE_SCALE,
            8
        )

        # =================================================
        # CROSS SECTION GRID
        # =================================================

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
# PROCESS SINGLE ARTERY
# =========================================================

def process_artery(
    artery_name,
    centerline_path,
    output_dir,
    ct,
    radius_map,
    case_name
):

    if centerline_path is None:

        print(
            f"{artery_name} centerline not found"
        )

        return

    centerline_vol, _, _ = load_nifti(
        centerline_path
    )

    centerline_points = (
        extract_centerline_points(
            centerline_vol
        )
    )

    print(
        f"{artery_name} raw points: "
        f"{len(centerline_points)}"
    )

    if len(centerline_points) < 5:

        print("Too few points")

        return

    straightened = (
        create_straightened_mpr_gpu(
            ct,
            centerline_points,
            radius_map
        )
    )

    print(
        f"\n{artery_name} output shape: "
        f"{straightened.shape}"
    )

    output_path = os.path.join(
        output_dir,
        f"{case_name}_{artery_name}_straightened.nii.gz"
    )

    save_nifti(
        straightened,
        np.eye(4),
        output_path
    )

    print("\nSaved:")
    print(output_path)


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

    case_name = match.group(1)

    print("\n=================================")
    print(f"Processing {case_name}")
    print("=================================")

    # =====================================================
    # FIND CENTERLINES AUTOMATICALLY
    # =====================================================

    artery1_candidates = glob.glob(
        os.path.join(
            CENTERLINE_DIR,
            f"{case_name}*artery_1_centerline.nii*"
        )
    )

    artery2_candidates = glob.glob(
        os.path.join(
            CENTERLINE_DIR,
            f"{case_name}*artery_2_centerline.nii*"
        )
    )

    artery1_centerline_path = (
        artery1_candidates[0]
        if len(artery1_candidates) > 0
        else None
    )

    artery2_centerline_path = (
        artery2_candidates[0]
        if len(artery2_candidates) > 0
        else None
    )

    radius_path = os.path.join(
        RADIUS_DIR,
        f"{case_name}_radius_map.nii.gz"
    )

    # =====================================================
    # LOAD CT
    # =====================================================

    ct, affine, header = load_nifti(
        ct_path
    )

    # =====================================================
    # LOAD RADIUS MAP
    # =====================================================

    if os.path.exists(radius_path):

        radius_map, _, _ = load_nifti(
            radius_path
        )

        print("Loaded shared radius map")

    else:

        print("Radius map missing")
        continue

    # =====================================================
    # PROCESS ARTERY 1
    # =====================================================

    print("\n========== ARTERY 1 ==========")

    process_artery(
        artery_name="artery_1",
        centerline_path=artery1_centerline_path,
        output_dir=ARTERY1_OUT_DIR,
        ct=ct,
        radius_map=radius_map,
        case_name=case_name
    )

    # =====================================================
    # PROCESS ARTERY 2
    # =====================================================

    print("\n========== ARTERY 2 ==========")

    process_artery(
        artery_name="artery_2",
        centerline_path=artery2_centerline_path,
        output_dir=ARTERY2_OUT_DIR,
        ct=ct,
        radius_map=radius_map,
        case_name=case_name
    )

print("\nDONE")