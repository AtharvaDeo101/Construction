import os
import sys
import io
import torch
import cv2
import numpy as np
import json
import gc
from PIL import Image
from depth_anything_3.api import DepthAnything3
import open3d as o3d

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"

DEFAULT_VIDEO_PATH = r"C:\Users\deoat\Desktop\Construct\assets\video_input\WhatsApp Video 2026-02-03 at 3.09.03 PM.mp4"
DEFAULT_OUTPUT_DIR = r"C:\Users\deoat\Desktop\Construct\data\scan_001"

FPS_EXTRACT = 2
IMG_SIZE = 518

# DA3 resolves depth AND pose in one coordinate frame per inference() call. All frames
# must go through in a single call or each group lands in its own frame at its own scale.
# If this OOMs, shrink PROCESS_RES / MODEL_REPO / FPS_EXTRACT -- never split the call.
PROCESS_RES = 336

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_FRAMES = 3
# DA3NESTED-GIANT-LARGE outputs depth already in metres, so these bounds are real metres.
MAX_DEPTH_METERS = 10.0

# DA3 confidence is unbounded and floored at 1.0 -- it is not a 0-1 probability, so any
# absolute threshold below 1.0 keeps every pixel. Cut by percentile instead.
# DA3's own exports default to 40, but ~30% of this scan's pixels sit at exactly 1.0
# (zero-confidence: textureless or out-of-frustum), so 40 barely bites. Measured on
# scan_001_fixed, 1-99% cloud extent by percentile: 0->14.8x6.3x8.2 m, 40->12.7x6.4x6.0,
# 60->8.0x6.3x5.3, 75->6.0x6.2x3.0. Raise this if the cloud still looks like fog.
CONFIDENCE_PERCENTILE = 60.0


def extract_frames(video_path: str, out_dir: str, fps: int = 2):
    """Extract frames from video at approximately `fps` and save to out_dir."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Info: {width}x{height} @ {video_fps:.2f} FPS, {total_frames} total frames")

    # Some WebM files (e.g. from MediaRecorder) report garbage FPS like 1000
    if video_fps <= 0 or video_fps > 240:
        print(f"Warning: suspicious FPS value {video_fps:.2f}, overriding to 30.0")
        video_fps = 30.0

    # Example: 30 FPS video, extract at 2 FPS -> take every 15th frame
    frame_interval = max(1, int(round(video_fps / fps)))

    count = 0
    saved_count = 0
    frame_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames at calculated interval
        if count % frame_interval == 0:
            frame_name = f"{saved_count:05d}.jpg"
            out_path = os.path.join(out_dir, frame_name)
            # Save with 95% JPEG quality
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(out_path)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {out_dir}")

    # Validate minimum frame count
    if saved_count < MIN_FRAMES:
        raise ValueError(
            f"Only {saved_count} frames extracted. Need at least {MIN_FRAMES} for multi-view consistency. "
            f"Try: longer video, higher FPS_EXTRACT, or slower camera movement."
        )

    return frame_paths


def run_da3_pipeline(image_paths, output_root):
    """Run DepthAnything3 on extracted frames: depth + poses + visualizations."""
    images_dir = os.path.join(output_root, "images")
    depth_dir = os.path.join(output_root, "depth")
    viz_dir = os.path.join(output_root, "viz")

    for d in [images_dir, depth_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy / symlink images to output_root/images
    rel_frame_paths = []
    for i, src_path in enumerate(image_paths):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(images_dir, filename)

        if not os.path.exists(dst_path):
            try:
                # Try symlink first (faster)
                os.symlink(os.path.abspath(src_path), dst_path)
            except (OSError, NotImplementedError):
                # Fallback to copy on Windows
                import shutil

                shutil.copy(src_path, dst_path)

        rel_frame_paths.append(f"images/{filename}")

    print("\n" + "=" * 60)
    print(f"Loading {MODEL_REPO}...")
    print(f"Device: {DEVICE} | Process res: {PROCESS_RES}")
    print(f"Single-call inference over all {len(image_paths)} frames")
    print("=" * 60 + "\n")

    model = (
        DepthAnything3.from_pretrained(
            MODEL_REPO,
            local_files_only=False,
        )
        .to(DEVICE)
        .eval()
    )

    images = [Image.open(p).convert("RGB") for p in image_paths]

    with torch.no_grad():
        prediction = model.inference(images, process_res=PROCESS_RES)

    depths = prediction.depth  # [N, H, W] float32
    extrinsics = prediction.extrinsics  # [N, 3, 4]
    intrinsics = prediction.intrinsics  # [N, 3, 3]

    has_confidence = hasattr(prediction, "conf")
    if has_confidence:
        confidences = prediction.conf  # [N, H, W]

    del prediction, images
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # A split call anchors one camera per group at the origin, so several cameras sitting
    # exactly there means the poses are in unrelated frames. (Zero is normal: the
    # reference view is normalised but not pinned to identity in the output frame.)
    cam_centers = np.stack(
        [np.linalg.inv(np.vstack([e, [0, 0, 0, 1]]))[:3, 3] for e in extrinsics]
    )
    n_at_origin = int((np.linalg.norm(cam_centers, axis=1) < 0.01).sum())
    if n_at_origin > 1:
        print(f"\nWARNING: {n_at_origin} cameras sit at the origin (expected 1).")
        print("  Poses are not in a shared coordinate frame -- do not trust this cloud.\n")

    # Path length beats bounding-box extent here: a walkthrough that returns near its
    # start has a small extent but a long path, and overlaid batches inflate extent.
    path_length = float(np.linalg.norm(np.diff(cam_centers, axis=0), axis=1).sum())

    if path_length < 0.1:  # Less than 10cm of travel
        print("\nWARNING: Camera barely moves!")
        print("  This may indicate:")
        print("   - Video captured from a fixed tripod (needs translation)")
        print("   - DA3 failed to estimate motion (try more distinctive scene features)")
        print(f"  Path length: {path_length * 100:.1f} cm\n")
    else:
        # Nested model outputs metres, so this is checkable: path/duration should look
        # like a walking speed. Wildly off means the metric head mis-scaled the scene.
        print(f"Camera path length: {path_length:.2f} m")
        print(f"Scene depth: median {np.median(depths):.2f} m, max {depths.max():.2f} m")

    first_img = Image.open(image_paths[0])

    pose_data = {
        "camera_model": "PINHOLE",
        "width": first_img.width,
        "height": first_img.height,
        "frames": [],
    }

    print(f"\nSaving outputs for {len(image_paths)} frames...")

    for i in range(len(image_paths)):
        idx_str = f"{i:05d}"

        depth_map = depths[i]
        depth_map_clipped = np.clip(depth_map, 0, MAX_DEPTH_METERS)
        np.save(os.path.join(depth_dir, f"{idx_str}.npy"), depth_map_clipped)

        if has_confidence:
            conf_map = confidences[i]
            np.save(os.path.join(depth_dir, f"{idx_str}_conf.npy"), conf_map)

        valid_depths = depth_map_clipped[depth_map_clipped > 1e-6]
        if len(valid_depths) > 0:
            depth_max = np.percentile(valid_depths, 99.5)
        else:
            depth_max = MAX_DEPTH_METERS

        depth_norm = np.clip(depth_map_clipped / depth_max, 0, 1)
        depth_viz = (depth_norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)

        if has_confidence:
            alpha = (confidences[i] * 255).astype(np.uint8)
            colored = cv2.merge([colored[:, :, 0], colored[:, :, 1], colored[:, :, 2], alpha])

        cv2.imwrite(os.path.join(viz_dir, f"{idx_str}_depth.png"), colored)

        # DA3 extrinsics: w2c: [R | t], x_cam = R x_world + t
        w2c_3x4 = extrinsics[i]
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        c2w_4x4 = np.linalg.inv(w2c_4x4)

        pose_data["frames"].append(
            {
                "file_path": rel_frame_paths[i],
                "transform_matrix": c2w_4x4.tolist(),
                "intrinsic_matrix": intrinsics[i].tolist(),
                "depth_path": f"depth/{idx_str}.npy",
                "confidence_path": f"depth/{idx_str}_conf.npy" if has_confidence else None,
            }
        )

    json_path = os.path.join(output_root, "transforms.json")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_root}")
    print(f" -> {len(image_paths)} depth maps saved to depth/")
    print(f" -> {len(image_paths)} visualizations in viz/")
    print(" -> Camera poses saved to transforms.json")
    print("\nIMPORTANT: Check viz/*.png files!")


def generate_raw_pointcloud(output_dir: str):
    """
    Read transforms.json + depth maps, create raw fused point cloud.
    Saves to output_dir/pointcloud/raw_cloud.ply
    """
    print("\n" + "=" * 60)
    print("GENERATING RAW POINT CLOUD")
    print("=" * 60 + "\n")

    transforms_path = os.path.join(output_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    width = data["width"]
    height = data["height"]

    print(f"Processing {len(frames)} frames...")

    # One threshold for the whole scan, not per frame: a per-frame percentile would keep
    # the same fraction of a badly-blurred frame as of a sharp one.
    conf_paths = [f["confidence_path"] for f in frames if f.get("confidence_path")]
    conf_thresh = None
    if conf_paths:
        all_conf = np.concatenate(
            [np.load(os.path.join(output_dir, p)).ravel() for p in conf_paths]
        )
        conf_thresh = float(np.percentile(all_conf, CONFIDENCE_PERCENTILE))
        print(
            f"Confidence: range [{all_conf.min():.2f}, {all_conf.max():.2f}], "
            f"cutting below {conf_thresh:.3f} (drops lowest {CONFIDENCE_PERCENTILE:.0f}%)"
        )
        del all_conf

    all_points = []
    all_colors = []

    for idx, frame in enumerate(frames):
        # Load depth
        depth_path = os.path.join(output_dir, frame["depth_path"])
        depth_map = np.load(depth_path)

        # Load confidence if available
        conf_path = frame.get("confidence_path")
        if conf_path:
            conf_map = np.load(os.path.join(output_dir, conf_path))
        else:
            conf_map = None

        # Load image for colors
        img_path = os.path.join(output_dir, frame["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}, skipping frame {idx}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # Resize depth/confidence to match original image size if needed
        if depth_map.shape[0] != height or depth_map.shape[1] != width:
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            if conf_map is not None:
                conf_map = cv2.resize(conf_map, (width, height), interpolation=cv2.INTER_LINEAR)

        # Create valid mask
        valid_mask = (depth_map > 0.01) & (depth_map < MAX_DEPTH_METERS)
        if conf_map is not None and conf_thresh is not None:
            valid_mask &= conf_map >= conf_thresh

        # Pixel grid
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Apply mask
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        colors_valid = img_rgb[valid_mask]

        if len(depth_valid) == 0:
            print(f"Warning: No valid points in frame {idx}, skipping")
            continue

        # Backproject to camera coordinates
        intrinsic = np.array(frame["intrinsic_matrix"])
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        c2w = np.array(frame["transform_matrix"])
        points_cam_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world = (c2w @ points_cam_h.T).T[:, :3]

        all_points.append(points_world)
        all_colors.append(colors_valid)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(frames)} frames")

    if len(all_points) == 0:
        raise RuntimeError("No valid points generated from any frame!")

    print("\nMerging point clouds...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"Total points: {len(all_points):,}")

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    pointcloud_dir = os.path.join(output_dir, "pointcloud")
    os.makedirs(pointcloud_dir, exist_ok=True)

    output_path = os.path.join(pointcloud_dir, "raw_cloud.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"\nSaved raw point cloud: {output_path}")
    print(f"  Points: {len(all_points):,}")

    return output_path


def run_step1(video_path: str, output_dir: str, fps_extract: int = FPS_EXTRACT) -> None:
    """
    Run full Step 1 pipeline: extract frames from video, run DA3 depth+pose, write to output_dir.
    output_dir will contain transforms.json, images/, depth/, viz/, and pointcloud/raw_cloud.ply
    """
    temp_img_dir = os.path.join(output_dir, "images_temp")
    try:
        print("Extracting frames from video")
        frame_paths = extract_frames(video_path, temp_img_dir, fps=fps_extract)

        print("\nRunning DA3 inference")
        run_da3_pipeline(frame_paths, output_dir)

        print("\nGenerating raw point cloud")
        generate_raw_pointcloud(output_dir)

    finally:
        if os.path.exists(temp_img_dir):
            import shutil

            try:
                shutil.rmtree(temp_img_dir)
                print("\nCleanup complete")
            except (PermissionError, OSError) as e:
                print(f"\nWarning: could not delete temp image dir {temp_img_dir}: {e}")


if __name__ == "__main__":
    print("STEP 1: VIDEO TO DEPTH + POSES")

    # Usage:
    #   python step1_extract_and_process.py <video_path> <output_dir>
    if len(sys.argv) >= 3:
        video_path = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        video_path = DEFAULT_VIDEO_PATH
        output_dir = DEFAULT_OUTPUT_DIR
        print("Using DEFAULT_VIDEO_PATH and DEFAULT_OUTPUT_DIR")

    print(f"Video path: {video_path}")
    print(f"Output dir: {output_dir}")

    orientation_file = os.path.join(output_dir, "device_orientation.json")
    if os.path.exists(orientation_file):
        print(f"Found device orientation samples at: {orientation_file}")

    try:
        run_step1(video_path, output_dir)
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY ERROR")
        print("  Fix by shrinking, in this order:")
        print(f"   - PROCESS_RES (now {PROCESS_RES}) -> 392 or 336")
        print(f"   - MODEL_REPO (now {MODEL_REPO}) -> depth-anything/DA3-LARGE or DA3-BASE")
        print(f"   - FPS_EXTRACT (now {FPS_EXTRACT}) -> 1, for fewer frames")
        print("  Do NOT re-introduce mini-batching: it silently corrupts the poses.")
        import traceback

        traceback.print_exc()
        sys.exit(1)  # caller checks the exit code; swallowing this reports a failed scan as success
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
