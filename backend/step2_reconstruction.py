#to perform numpy like operations - GUP is used 
#to perform Open3D operation(mesh) - CPU is used (Open3D need Open3D repo to use cuda)

import os
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
import warnings

# CUDA/CuPy imports (CuPy works on Windows, cuML doesn't)
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print(f" CuPy GPU acceleration enabled")

except ImportError as e:
    CUDA_AVAILABLE = False
    print(" Running on CPU.")


# Check Open3D CUDA availability
o3d_cuda = o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available() else None
if o3d_cuda:
    print(f"Open3D CUDA device available: {o3d_cuda}")
else:
    print("Open3D CUDA not available")


DEFAULT_OUTPUT_DIR = os.getenv(
    "CONSTRUCT_OUTPUT_DIR",
    r"C:\Users\deoat\Desktop\Construct\data\scan_001"
)

RAW_POINTCLOUD_PATH = os.path.join(
    DEFAULT_OUTPUT_DIR, "pointcloud", "raw_cloud.ply"
)

DEPTH_TRUNC = 8.0
VOXEL_SIZE = 0.015
CONFIDENCE_THRESH = 0.8

DUPLICATE_EPS = 0.04
DUPLICATE_MIN_SAMPLES = 10

POISSON_DEPTH = 10
POISSON_SCALE = 1.1
POISSON_LINEAR_FIT = False
DENSITY_FILTER_QUANTILE = 0.05

FLOORPLAN_HEIGHT = 0.1
SLICE_THICKNESS = 0.08
FLOORPLAN_RESOLUTION = 1024

# draw_geometries() blocks on a native window, which a container has no display for --
# CONSTRUCT_SHOW_VIZ=0 turns it off (the Docker image sets it).
SHOW_VISUALIZATIONS = os.getenv("CONSTRUCT_SHOW_VIZ", "1") != "0"
VISUALIZATION_MODE = "web"

def to_gpu(arr):
    """Move numpy array to GPU if available"""
    if CUDA_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move array back to CPU"""
    if CUDA_AVAILABLE and hasattr(arr, 'get'):
        return cp.asnumpy(arr)
    return arr

def gpu_dbscan_hybrid(xy_points, eps=0.04, min_samples=10):
    """
    Hybrid GPU/CPU DBSCAN:
    - Computes distance operations on GPU (CuPy)
    - Runs clustering algorithm on CPU (sklearn)
    Works on Windows without cuML
    """
    n_points = len(xy_points)
    
    if not CUDA_AVAILABLE or n_points < 1000:
        print(f"  Running DBSCAN on CPU ({n_points:,} points)...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        return clustering.fit_predict(xy_points)
    
    print(f"  Running hybrid GPU/CPU DBSCAN ({n_points:,} points)...")
    print(f"    - Distance computation: GPU (CuPy)")
    print(f"    - Clustering algorithm: CPU (sklearn)")
    
    # Move data to GPU
    xy_gpu = cp.asarray(xy_points, dtype=cp.float32)
    

    chunk_size = min(5000, n_points)
    
    if n_points <= chunk_size:
        # Small dataset: compute full distance matrix
        diff = xy_gpu[:, cp.newaxis, :] - xy_gpu[cp.newaxis, :, :]
        dist_matrix_gpu = cp.sqrt(cp.sum(diff ** 2, axis=2))
        dist_matrix = cp.asnumpy(dist_matrix_gpu)
    else:
        # Large dataset: use radius neighbors (sparse approach)
        print(f"    - Using radius neighbors for memory efficiency...")
        from sklearn.neighbors import NearestNeighbors
        
        # Compute k-nearest neighbors on CPU (more memory efficient)
        nbrs = NearestNeighbors(radius=eps, algorithm='ball_tree', n_jobs=-1)
        nbrs.fit(xy_points)
        
        # Get neighbors within radius
        distances, indices = nbrs.radius_neighbors(xy_points)
        
        # sklearn DBSCAN with precomputed sparse neighbors
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        
        # Create sparse distance matrix
        from scipy.sparse import lil_matrix
        dist_sparse = lil_matrix((n_points, n_points))
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            dist_sparse[i, idxs] = dists
        
        return clustering.fit_predict(dist_sparse)
    
    # Run clustering on CPU with precomputed distances
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    
    return labels

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def load_raw_pointcloud(pcd_path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Raw point cloud not found: {pcd_path}")

    print(f"\nLoading raw point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded point cloud with {len(pcd.points):,} points")
    return pcd

def downsample_and_filter_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    print(f"\nDownsampling with voxel size {VOXEL_SIZE} m...")
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Points after voxel downsample: {len(pcd_ds.points):,}")

    print("Applying statistical outlier removal...")
    pcd_stat, ind = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Points after statistical removal: {len(pcd_stat.points):,}")

    print("Applying radius outlier removal...")
    pcd_rad, ind = pcd_stat.remove_radius_outlier(nb_points=16, radius=0.08)
    print(f"Points after radius removal: {len(pcd_rad.points):,}")

    if SHOW_VISUALIZATIONS:
        print("Visualizing filtered point cloud...")
        o3d.visualization.draw_geometries([pcd_rad], window_name="Filtered point cloud")

    return pcd_rad


def visualize_model(geometry, window_name="3D Model"):
    """Smart visualization dispatcher"""
    if VISUALIZATION_MODE == "web":
        o3d.visualization.draw(geometry, title=window_name)
    elif VISUALIZATION_MODE == "native":
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1920, height=1080)
        vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = False
        vis.run()
        vis.destroy_window()
    elif VISUALIZATION_MODE == "export":
        export_path = os.path.join(DEFAULT_OUTPUT_DIR, "mesh", "mesh_for_viewer.ply")
        ensure_dir(os.path.dirname(export_path))
        o3d.io.write_triangle_mesh(export_path, geometry)
        print(f"\n✓ Exported mesh: {export_path}")

# def remove_duplicate_structures(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
#     """GPU-accelerated duplicate structure removal"""
#     print("\nRemoving duplicate structures with DBSCAN clustering...")
#     pts = np.asarray(pcd.points)
#     cols = np.asarray(pcd.colors) if pcd.has_colors() else None

#     if len(pts) == 0:
#         print("Point cloud is empty, skipping duplicate removal.")
#         return pcd

#     if len(pts) < 1000:
#         print(f"Only {len(pts)} points - skipping duplicate removal to preserve data.")
#         return pcd

#     xy = pts[:, :2].astype(np.float32)
    
#     labels = gpu_dbscan_hybrid(xy, eps=DUPLICATE_EPS, min_samples=DUPLICATE_MIN_SAMPLES)

#     unique_labels = set(labels)
#     num_noise = np.sum(labels == -1)
#     if -1 in unique_labels:
#         unique_labels.remove(-1)

#     print(f"Found {len(unique_labels)} dense clusters, {num_noise} noise points")

#     if len(unique_labels) > 0 and num_noise < len(pts) * 0.2:
#         print(f"Warning: Only {num_noise}/{len(pts)} noise points - clustering too aggressive, skipping.")
#         return pcd

#     mask_noise = labels == -1
#     new_points = [pts[mask_noise]]
#     new_colors = [cols[mask_noise]] if cols is not None else None

#     for lbl in unique_labels:
#         mask = labels == lbl
#         cluster_points = pts[mask]
#         if cluster_points.shape[0] == 0:
#             continue

#         sample_rate = max(1, cluster_points.shape[0] // 50)
#         sampled = cluster_points[::sample_rate]
        
#         new_points.append(sampled)
#         if new_colors is not None and cols is not None:
#             sampled_colors = cols[mask][::sample_rate]
#             new_colors.append(sampled_colors)

#     new_points = np.vstack(new_points) if len(new_points) > 0 else np.empty((0, 3))
#     if cols is not None and new_colors is not None:
#         new_colors = np.vstack(new_colors)
#     else:
#         new_colors = None

#     print(f"Points after duplicate handling: {len(new_points):,}")

#     if len(new_points) < len(pts) * 0.05:
#         print(f"Warning: Only {len(new_points)} points remain - reverting to original cloud.")
#         return pcd

#     new_pcd = o3d.geometry.PointCloud()
#     new_pcd.points = o3d.utility.Vector3dVector(new_points)
#     if new_colors is not None:
#         new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

#     if SHOW_VISUALIZATIONS:
#         visualize_model(new_pcd, "De-duplicated Point Cloud")

#     return new_pcd

# def reconstruct_mesh_poisson(pcd: o3d.geometry.PointCloud, output_dir: str) -> o3d.geometry.TriangleMesh:
#     print("\n" + "=" * 60)
#     print("GENERATING MESH (Poisson)")
#     print("=" * 60)

#     if len(pcd.points) == 0:
#         raise RuntimeError("Point cloud is empty, cannot reconstruct mesh.")

#     print("Estimating normals...")
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
#     )
#     pcd.orient_normals_consistent_tangent_plane(30)

#     print(f"Running Poisson surface reconstruction (depth={POISSON_DEPTH})...")
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd,
#         depth=POISSON_DEPTH,
#         scale=POISSON_SCALE,
#         linear_fit=POISSON_LINEAR_FIT,
#     )

#     densities = np.asarray(densities)
#     print("Filtering low-density vertices...")
    
#     if CUDA_AVAILABLE:
#         densities_gpu = to_gpu(densities)
#         density_thresh = float(cp.quantile(densities_gpu, DENSITY_FILTER_QUANTILE))
#     else:
#         density_thresh = np.quantile(densities, DENSITY_FILTER_QUANTILE)
    
#     vertices_to_remove = densities < density_thresh
#     mesh.remove_vertices_by_mask(vertices_to_remove)

#     print("Cropping mesh to point cloud bounding box...")
#     bbox = pcd.get_axis_aligned_bounding_box()
#     bbox = bbox.scale(1.05, bbox.get_center())
#     mesh = mesh.crop(bbox)

#     mesh.compute_vertex_normals()

#     ensure_dir(os.path.join(output_dir, "mesh"))
#     mesh_path = os.path.join(output_dir, "mesh", "mesh_poisson.ply")
#     o3d.io.write_triangle_mesh(mesh_path, mesh)
#     print(f"Saved mesh to: {mesh_path}")

#     if SHOW_VISUALIZATIONS:
#         visualize_model(mesh, "Reconstructed Mesh")

#     return mesh

# def generate_floor_plan(pcd: o3d.geometry.PointCloud, output_dir: str):
#     print("\n" + "=" * 60)
#     print("GENERATING FLOOR PLAN")
#     print("=" * 60)

#     pts = np.asarray(pcd.points)
#     if pts.shape[0] == 0:
#         print("Point cloud empty, skipping floor plan.")
#         return

#     if CUDA_AVAILABLE:
#         z_gpu = to_gpu(pts[:, 2])
#         floor_z = float(cp.percentile(z_gpu, 5))
#     else:
#         z_values = pts[:, 2]
#         floor_z = np.percentile(z_values, 5)

#     slice_height = floor_z + FLOORPLAN_HEIGHT
#     print(f"Slicing at {slice_height:.2f} m (floor ~ {floor_z:.2f} m)")

#     mask = np.abs(pts[:, 2] - slice_height) <= SLICE_THICKNESS
#     slice_points = pts[mask]

#     print(f"Extracted {len(slice_points)} points in slice")

#     if slice_points.shape[0] == 0:
#         print("No points in slice; skipping floor plan image generation.")
#         return

#     xs = slice_points[:, 0]
#     ys = slice_points[:, 1]

#     x_min, x_max = xs.min(), xs.max()
#     y_min, y_max = ys.min(), ys.max()

#     pad = 0.05 * max(x_max - x_min, y_max - y_min)
#     x_min -= pad
#     x_max += pad
#     y_min -= pad
#     y_max += pad

#     width = height = FLOORPLAN_RESOLUTION
#     img = np.zeros((height, width, 1), dtype=np.uint8)

#     x_norm = (xs - x_min) / (x_max - x_min + 1e-8)
#     y_norm = (ys - y_min) / (y_max - y_min + 1e-8)

#     u = (x_norm * (width - 1)).astype(np.int32)
#     v = (1.0 - y_norm) * (height - 1)
#     v = v.astype(np.int32)

#     img[v, u, 0] = 255

#     floorplan_dir = os.path.join(output_dir, "floorplan")
#     ensure_dir(floorplan_dir)
#     floorplan_path = os.path.join(floorplan_dir, "floorplan.png")
#     cv2.imwrite(floorplan_path, img)
#     print(f"Saved floor plan image: {floorplan_path}")

#     if SHOW_VISUALIZATIONS:
#         cv2.imshow("Floor Plan", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

def main():
    output_dir = DEFAULT_OUTPUT_DIR

    print("=" * 60)
    print("STEP 2: RECONSTRUCTION (GPU-ACCELERATED)")
    print("=" * 60)
    
    if CUDA_AVAILABLE:
        print(f"Running with hybrid GPU/CPU acceleration")
    else:
        print("Running on CPU only")

    ensure_dir(output_dir)

    pcd_raw = load_raw_pointcloud(RAW_POINTCLOUD_PATH)
    pcd_filtered = downsample_and_filter_pointcloud(pcd_raw)
    # pcd_dedup = remove_duplicate_structures(pcd_filtered)
    # mesh = reconstruct_mesh_poisson(pcd_dedup, output_dir)
    # generate_floor_plan(pcd_dedup, output_dir)

    print("\nSTEP 2 COMPLETE")
    print(f"Outputs written under: {output_dir}")

if __name__ == "__main__":
    main()


