
import os 
import torch
import cv2
import numpy as np 
import json 
import gc
from PIL import Image
from depth_anything_3.api import DepthAnything3


MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"  


VIDEO_PATH = r"C:\Users\deoat\Desktop\Construct\assets\video_input\WhatsApp Video 2026-01-27 at 3.24.44 PM.mp4"
OUTPUT_DIR = r"C:\Users\deoat\Desktop\Construct\data\scan_001"
FPS_EXTRACT = 2          
IMG_SIZE = 518          
MINI_BATCH_SIZE = 3      

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_FRAMES = 3           
MAX_DEPTH_METERS = 10.0  


def extract_frames(video_path, out_dir, fps=2):

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
    depth_dir  = os.path.join(output_root, "depth")
    viz_dir    = os.path.join(output_root, "viz")

    for d in [images_dir, depth_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    
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
    
    model = DepthAnything3.from_pretrained(MODEL_REPO,local_files_only=True).to(DEVICE).eval()
    
    
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
                # input_size=IMG_SIZE,             
                # max_batch_size=len(batch_images), 
            )
        
        all_depths.append(prediction.depth)
        all_extrinsics.append(prediction.extrinsics)
        all_intrinsics.append(prediction.intrinsics)
        
        # Check for confidence maps
        if hasattr(prediction, 'conf'):
            all_confidences.append(prediction.conf)
        
        del prediction
        del batch_images
        torch.cuda.empty_cache()
        gc.collect()
    
    
    print("\nMerging all batches...")
    depths = np.concatenate(all_depths, axis=0)
    extrinsics = np.concatenate(all_extrinsics, axis=0)
    intrinsics = np.concatenate(all_intrinsics, axis=0)
    
    has_confidence = len(all_confidences) > 0
    if has_confidence:
        confidences = np.concatenate(all_confidences, axis=0)
    
    
    cam_positions = extrinsics[:, :3, 3]  # Extract translation vectors
    movement_range = np.linalg.norm(cam_positions.max(axis=0) - cam_positions.min(axis=0))

    if movement_range < 0.1:  # Less than 10cm total movement
        print("\n  WARNING: Camera positions are nearly identical!")
        print("    This may indicate:")
        print("    - Video captured from a fixed tripod (needs translation)")
        print("    - DA3 failed to estimate motion (try more distinctive scene features)")
        print(f"    Movement detected: {movement_range*100:.1f} cm\n")
    else:
        print(f"Camera movement detected: {movement_range:.2f} meters")

    
    first_img = Image.open(image_paths[0])
    
    pose_data = {
        "camera_model": "PINHOLE",
        "width":  first_img.width,
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
            colored = cv2.merge([colored[:,:,0], colored[:,:,1], colored[:,:,2], alpha])
        
        cv2.imwrite(os.path.join(viz_dir, f"{idx_str}_depth.png"), colored)

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
    print(f"✓ Processing Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_root}")
    print(f"  → {len(image_paths)} depth maps saved to depth/")
    print(f"  → {len(image_paths)} visualizations in viz/")
    print(f"  → Camera poses saved to transforms.json")
    print(f"\n  IMPORTANT: Check viz/*.png files!")


if __name__ == "__main__":
    print(f"# STEP 1: VIDEO TO DEPTH + POSES")
 
    
    temp_img_dir = os.path.join(OUTPUT_DIR, "images_temp")
    
    try:
        print("Extracting frames from video")
        frame_paths = extract_frames(VIDEO_PATH, temp_img_dir, fps=FPS_EXTRACT)
        
        print("\n Running DA3 inference")
        run_da3_pipeline(frame_paths, OUTPUT_DIR)
        
        print("\nCleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_img_dir)
        print("Temporary files removed")
        
    except torch.cuda.OutOfMemoryError:
        print(f"\nCUDA OUT OF MEMORY ERROR")
    
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
