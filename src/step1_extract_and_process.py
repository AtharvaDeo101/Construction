
import os
import torch
import cv2
import numpy as np
import json
import gc
from PIL import Image
from depth_anything_3.api import DepthAnything3
import open3d as o3d

MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"

# Defaults when run as script (overridable via run_pipeline)
DEFAULT_VIDEO_PATH = r"C:\Users\deoat\Desktop\Construct\assets\video_input\room.mp4"
DEFAULT_OUTPUT_DIR = r"C:\\Users\\deoat\\Desktop\\Construct\\data\\scan_001"
FPS_EXTRACT = 2
IMG_SIZE = 518
MINI_BATCH_SIZE = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_FRAMES = 3
MAX_DEPTH_METERS = 10.0


def extract_frames(video_path: str, out_dir: str, fps: int = 2):
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

    # Example: 30 FPS video, extract at 2 FPS → take every 15th frame
    frame_interval = max(1, int(video_fps / fps))

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
    print(f"✓ Extracted {saved_count} frames to {out_dir}")

    # Validate minimum frame count
    if saved_count < MIN_FRAMES:
        raise ValueError(
            f"Only {saved_count} frames extracted. Need ≥{MIN_FRAMES} for multi-view consistency. "
            f"Try: longer video, higher FPS_EXTRACT, or slower camera movement."
        )

    return frame_paths


def run_da3_pipeline(image_paths, output_root):

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

    print(f"\n{'='*60}")
    print(f"Loading {MODEL_REPO}...")
    print(f"Device: {DEVICE} | Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Mini-batch size: {MINI_BATCH_SIZE} frames")
    print(f"{'='*60}\n")

    model = DepthAnything3.from_pretrained(
        MODEL_REPO,
        local_files_only=False
    ).to(DEVICE).eval()

    all_depths = []
    all_extrinsics = []
    all_intrinsics = []
    all_confidences = []

    num_batches = (len(image_paths) + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE

    print(f"Processing {len(image_paths)} frames in {num_batches} mini-batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * MINI_BATCH_SIZE
        end_idx = min(start_idx + MINI_BATCH_SIZE, len(image_paths))

        batch_paths = image_paths[start_idx:end_idx]
        print(f"  Batch {batch_idx+1}/{num_batches}: frames {start_idx} to {end_idx-1}")

        # Load mini-batch as PIL RGB
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        # Run inference on this mini-batch
        with torch.no_grad():
            prediction = model.inference(
                batch_images,
                # input_size=IMG_SIZE,  # can be tuned if needed
            )

        all_depths.append(prediction.depth)          # [N, H, W] float32
        all_extrinsics.append(prediction.extrinsics) # [N, 3, 4]
        all_intrinsics.append(prediction.intrinsics) # [N, 3, 3]
        if hasattr(prediction, 'conf'):
            all_confidences.append(prediction.conf)  # [N, H, W]

        del prediction
        del batch_images
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("\nMerging all batches...")
    depths = np.concatenate(all_depths, axis=0)
    extrinsics = np.concatenate(all_extrinsics, axis=0)
    intrinsics = np.concatenate(all_intrinsics, axis=0)

    has_confidence = len(all_confidences) > 0
    if has_confidence:
        confidences = np.concatenate(all_confidences, axis=0)

    # Check approximate camera movement magnitude
    cam_positions = extrinsics[:, :3, 3]  # Extract translation vectors (w2c)
    movement_range = np.linalg.norm(cam_positions.max(axis=0) - cam_positions.min(axis=0))

    if movement_range < 0.1:  # Less than 10cm total movement
        print("\n WARNING: Camera positions are nearly identical!")
        print("  This may indicate:")
        print("   - Video captured from a fixed tripod (needs translation)")
        print("   - DA3 failed to estimate motion (try more distinctive scene features)")
        print(f"  Movement detected: {movement_range*100:.1f} cm\n")
    else:
        print(f"Camera movement detected (w2c translations): {movement_range:.2f} meters")

    first_img = Image.open(image_paths[0])

    pose_data = {
        "camera_model": "PINHOLE",
        "width": first_img.width,
        "height": first_img.height,
        "frames": []
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

        # DA3 extrinsics: OpenCV/Colmap w2c: [R | t], x_cam = R x_world + t
        w2c_3x4 = extrinsics[i]
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        c2w_4x4 = np.linalg.inv(w2c_4x4)

        pose_data["frames"].append({
            "file_path": rel_frame_paths[i],
            "transform_matrix": c2w_4x4.tolist(),
            "intrinsic_matrix": intrinsics[i].tolist(),
            "depth_path": f"depth/{idx_str}.npy",
            "confidence_path": f"depth/{idx_str}_conf.npy" if has_confidence else None
        })

    json_path = os.path.join(output_root, "transforms.json")
    with open(json_path, 'w') as f:
        json.dump(pose_data, f, indent=2)

    print(f"\n{'='*60}")
    print("✓ Processing Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_root}")
    print(f" → {len(image_paths)} depth maps saved to depth/")
    print(f" → {len(image_paths)} visualizations in viz/")
    print(" → Camera poses saved to transforms.json")
    print("\n IMPORTANT: Check viz/*.png files!")



def generate_raw_pointcloud(output_dir: str):
    """
    Read transforms.json + depth maps, create raw fused point cloud.
    Saves to output_dir/pointcloud/raw_cloud.ply
    """
    print(f"\n{'='*60}")
    print("GENERATING RAW POINT CLOUD")
    print(f"{'='*60}\n")
    
    transforms_path = os.path.join(output_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    frames = data["frames"]
    width = data["width"]
    height = data["height"]
    
    print(f"Processing {len(frames)} frames...")
    
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
        
        # CRITICAL FIX: Resize depth and confidence to match original image size
        if depth_map.shape[0] != height or depth_map.shape[1] != width:
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            if conf_map is not None:
                conf_map = cv2.resize(conf_map, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Create valid mask
        if conf_map is not None:
            valid_mask = (conf_map > 0.5) & (depth_map > 0.01) & (depth_map < MAX_DEPTH_METERS)
        else:
            valid_mask = (depth_map > 0.01) & (depth_map < MAX_DEPTH_METERS)
        
        # Get camera matrices
        c2w = np.array(frame["transform_matrix"])
        intrinsic = np.array(frame["intrinsic_matrix"])
        
        # Create pixel grid
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Apply valid mask
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        colors_valid = img_rgb[valid_mask]
        
        if len(depth_valid) == 0:
            print(f"Warning: No valid points in frame {idx}, skipping")
            continue
        
        # Backproject to camera coordinates
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        points_cam_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world = (c2w @ points_cam_h.T).T[:, :3]
        
        all_points.append(points_world)
        all_colors.append(colors_valid)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(frames)} frames")
    
    if len(all_points) == 0:
        raise RuntimeError("No valid points generated from any frame!")
    
    print("\nMerging point clouds...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    print(f"Total points: {len(all_points):,}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save
    pointcloud_dir = os.path.join(output_dir, "pointcloud")
    os.makedirs(pointcloud_dir, exist_ok=True)
    
    output_path = os.path.join(pointcloud_dir, "raw_cloud.ply")
    o3d.io.write_point_cloud(output_path, pcd)
    
    print(f"\n✓ Saved raw point cloud: {output_path}")
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
            shutil.rmtree(temp_img_dir)
            print("\n✓ Cleanup complete")



if __name__ == "__main__":
    print("# STEP 1: VIDEO TO DEPTH + POSES")
    try:
        run_step1(DEFAULT_VIDEO_PATH, DEFAULT_OUTPUT_DIR)
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY ERROR")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
