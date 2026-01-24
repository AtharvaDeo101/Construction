import os 
import torch
import cv2
import numpy as np 
import json 
from PIL import Image
from depth_anything_3.api import DepthAnything3


MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE" 


VIDEO_PATH = "../assets/video_input/room_scan.mp4"
OUTPUT_DIR = "../data/scan_001"
FPS_EXTRACT = 2          # Frames per second to extract (2-5 recommended for indoor)
IMG_SIZE = 518           # Input resolution (multiple of 14, DA3 optimal)
MAX_BATCH_SIZE = 4       # VRAM-dependent: 1-2 for 12GB, 4-8 for 24GB+

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_FRAMES = 3           # Minimum frames needed for multi-view consistency
MAX_DEPTH_METERS = 10.0  # Clip outlier depths (typical indoor max ~8m)




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

    """read video frame by frame, and selectively save only some of the frames to disk at a desired sampling rate"""

    # Calculate sampling interval
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

            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(out_path)
            saved_count += 1
        count += 1

    cap.release()
    print(f"✓ Extracted {saved_count} frames to {out_dir}")
    
    # Validate minimum frame count for multi-view processing
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
                
                os.symlink(os.path.abspath(src_path), dst_path)
            except (OSError, NotImplementedError):

                import shutil
                shutil.copy(src_path, dst_path)
        
        rel_frame_paths.append(f"images/{filename}")
    
    
    #  Load DA3 Model 
    print(f"\n{'='*60}")
    print(f"Loading {MODEL_REPO}...")
    print(f"Device: {DEVICE} | Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'='*60}\n")
    
    # Downloads model from Hugging Face on first run (auto-cached)
    model = DepthAnything3.from_pretrained(MODEL_REPO).to(DEVICE).eval()

    print("Loading frames into memory...")
    images = [Image.open(p).convert("RGB") for p in image_paths]





    with torch.no_grad():
        prediction = model.inference(
            images,
            input_size=IMG_SIZE,           
            max_batch_size=MAX_BATCH_SIZE  
        )


    depths     = prediction.depth.cpu().numpy()
    extrinsics = prediction.extrinsics.cpu().numpy() 

    intrinsics = prediction.intrinsics.cpu().numpy()
    
#confidence mapping
    has_confidence = hasattr(prediction, 'conf')
    if has_confidence:
        confidences = prediction.conf.cpu().numpy()

    cam_positions = extrinsics[:, :3, 3]  # Extract camera prosition from real world
    movement_range = np.linalg.norm(cam_positions.max(axis=0) - cam_positions.min(axis=0))


    if movement_range < 0.1:  # Less than 10cm total movement
        print("⚠️  WARNING: Camera positions are nearly identical!")
        print("    This may indicate:")
        print("    - Video captured from a fixed tripod (needs translation)")
        print("    - DA3 failed to estimate motion (try more distinctive scene features)")
        print(f"    Movement detected: {movement_range*100:.1f} cm\n")

#PREPARING OUTPUT

    pose_data = {
        "camera_model": "PINHOLE",                 # Standard pinhole projection
        "width":  images[0].width,
        "height": images[0].height,
        "frames": []
    }


#PROCESSING EACH FRAME ONE BY ONE 
    print(f"Saving outputs...")
    for i in range(len(images)):
        idx_str = f"{i:05d}"

        depth_map = depths[i]
        
        # Clip extreme outliers (can occur from reflections/windows)
        depth_map_clipped = np.clip(depth_map, 0, MAX_DEPTH_METERS)
        np.save(os.path.join(depth_dir, f"{idx_str}.npy"), depth_map_clipped)


        if has_confidence:  #saving confidence
            conf_map = confidences[i]
            np.save(os.path.join(depth_dir, f"{idx_str}_conf.npy"), conf_map)

        valid_depths = depth_map_clipped[depth_map_clipped > 1e-6]  # Ignore zero/invalid
        if len(valid_depths) > 0:
            depth_max = np.percentile(valid_depths, 99.5)
        else:
            depth_max = MAX_DEPTH_METERS
        
        # Normalize to 0-1 range
        depth_norm = np.clip(depth_map_clipped / depth_max, 0, 1)
        
        # Convert to 8-bit and apply TURBO colormap (red=near, blue=far)
        depth_viz = (depth_norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
        
        # Optionally overlay confidence as transparency (if available)
        if has_confidence:
            alpha = (confidences[i] * 255).astype(np.uint8)
            colored = cv2.merge([colored[:,:,0], colored[:,:,1], colored[:,:,2], alpha])
        
        cv2.imwrite(os.path.join(viz_dir, f"{idx_str}_depth.png"), colored)


        # DA3 outputs world-to-camera [3,4] matrices
        # Most 3D tools expect camera-to-world [4,4] transforms
        
        w2c_3x4 = extrinsics[i]  # [3, 4] matrix [R|t]
        
        # Build full 4x4 homogeneous matrix
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        # Last row stays [0, 0, 0, 1]
        
        # Invert to get camera-to-world
        c2w_4x4 = np.linalg.inv(w2c_4x4)



#STORING FRAME META DATA 
    pose_data["frames"].append({
            "file_path": rel_frame_paths[i],          # Relative path to RGB image
            "transform_matrix": c2w_4x4.tolist(),     # 4x4 c2w as nested list
            "intrinsic_matrix": intrinsics[i].tolist(), # 3x3 camera matrix K
            "depth_path": f"depth/{idx_str}.npy",     # Link to depth file
            "confidence_path": f"depth/{idx_str}_conf.npy" if has_confidence else None
        })
    

    json_path = os.path.join(output_root, "transforms.json")
    with open(json_path, 'w') as f:
        json.dump(pose_data, f, indent=2)







if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# STEP 1: VIDEO TO DEPTH + POSES")
    print(f"{'#'*60}\n")
    
    # Create temporary extraction directory 
    temp_img_dir = os.path.join(OUTPUT_DIR, "images_temp")
    
    try:
        # Phase 1: Extract frames from video
        frame_paths = extract_frames(VIDEO_PATH, temp_img_dir, fps=FPS_EXTRACT)
        
        # Phase 2: Run DA3 depth+pose estimation
        run_da3_pipeline(frame_paths, OUTPUT_DIR)
        
        # Optional: Clean up temporary extraction folder (frames now in images/)
        # import shutil
        # shutil.rmtree(temp_img_dir)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()