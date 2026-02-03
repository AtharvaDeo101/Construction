
import os
import json
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN


DEFAULT_OUTPUT_DIR = r"C:\Users\deoat\Desktop\Construct\data\scan_001"  
RAW_POINTCLOUD_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "pointcloud", "raw_cloud.ply")


# Point cloud filtering
DEPTH_TRUNC = 8.0          # Keep points within this distance (meters)
VOXEL_SIZE = 0.015         # Smaller = denser cloud, larger = smoother
CONFIDENCE_THRESH = 0.6    # You can raise to 0.7–0.8 to reduce duplicates, or lower to 0.4–0.5 if you see missing chunks

# Duplicate structure removal (DBSCAN)
DUPLICATE_EPS = 0.04       # Meters in XY space for clustering overlapping walls
DUPLICATE_MIN_SAMPLES = 10 # Minimum points to form a cluster

# Poisson reconstruction
POISSON_DEPTH = 10         # Higher = more detail (but can create more ghosts)
POISSON_SCALE = 1.1
POISSON_LINEAR_FIT = False
DENSITY_FILTER_QUANTILE = 0.05  # Remove lowest X% of vertices by density

# Floor plan generation
FLOORPLAN_HEIGHT = 0.1     # Height above detected floor (meters)
SLICE_THICKNESS = 0.08     # Thickness of slice around that height (meters)
FLOORPLAN_RESOLUTION = 1024  # Output image size (pixels)

SHOW_VISUALIZATIONS = True


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)




def load_raw_pointcloud(pcd_path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Raw point cloud not found: {pcd_path}")

    print(f"\nLoading raw point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded point cloud with {len(pcd.points):,} points")

    # Optional depth truncation along camera Z if you saved points in camera coordinates
    # Here we assume world coordinates, so we skip explicit truncation

    return pcd


def downsample_and_filter_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    print(f"\nDownsampling with voxel size {VOXEL_SIZE} m...")
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Points after voxel downsample: {len(pcd_ds.points):,}")

    # Statistical outlier removal
    print("Applying statistical outlier removal...")
    pcd_stat, ind = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Points after statistical removal: {len(pcd_stat.points):,}")

    # Radius outlier removal (helps remove small floating clusters)
    print("Applying radius outlier removal...")
    pcd_rad, ind = pcd_stat.remove_radius_outlier(nb_points=16, radius=0.08)
    print(f"Points after radius removal: {len(pcd_rad.points):,}")

    if SHOW_VISUALIZATIONS:
        print("Visualizing filtered point cloud...")
        o3d.visualization.draw_geometries([pcd_rad], window_name="Filtered point cloud")

    return pcd_rad



def remove_duplicate_structures(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Use DBSCAN in XY space to detect very dense overlapping structures (e.g., multiple copies
    of the same wall) and collapse each cluster to a single averaged representative.
    """
    print("\nRemoving duplicate structures with DBSCAN clustering...")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    if len(pts) == 0:
        print("Point cloud is empty, skipping duplicate removal.")
        return pcd

    # SAFETY: Only apply duplicate removal if we have enough points
    if len(pts) < 1000:
        print(f"Only {len(pts)} points - skipping duplicate removal to preserve data.")
        return pcd

    xy = pts[:, :2]  # Cluster in XY plane
    clustering = DBSCAN(eps=DUPLICATE_EPS, min_samples=DUPLICATE_MIN_SAMPLES).fit(xy)
    labels = clustering.labels_

    unique_labels = set(labels)
    num_noise = np.sum(labels == -1)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # -1 is noise

    print(f"Found {len(unique_labels)} dense clusters, {num_noise} noise points")

    # SAFETY: If clustering marked too many points as clusters (> 80%), skip it
    if len(unique_labels) > 0 and num_noise < len(pts) * 0.2:
        print(f"Warning: Only {num_noise}/{len(pts)} noise points - clustering too aggressive, skipping.")
        return pcd

    # Keep noise points as they are
    mask_noise = labels == -1
    new_points = [pts[mask_noise]]
    new_colors = [cols[mask_noise]] if cols is not None else None

    # For each cluster, keep all points (not just one averaged point)
    # This prevents excessive data loss
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_points = pts[mask]
        if cluster_points.shape[0] == 0:
            continue

        # Instead of collapsing to 1 point, keep a representative sample
        # Take every Nth point from the cluster to preserve structure
        sample_rate = max(1, cluster_points.shape[0] // 50)  # Keep ~50 points per cluster
        sampled = cluster_points[::sample_rate]
        
        new_points.append(sampled)
        if new_colors is not None and cols is not None:
            sampled_colors = cols[mask][::sample_rate]
            new_colors.append(sampled_colors)

    new_points = np.vstack(new_points) if len(new_points) > 0 else np.empty((0, 3))
    if cols is not None and new_colors is not None:
        new_colors = np.vstack(new_colors)
    else:
        new_colors = None

    print(f"Points after duplicate handling: {len(new_points):,}")

    # SAFETY: If we lost too many points (> 95%), return original
    if len(new_points) < len(pts) * 0.05:
        print(f"Warning: Only {len(new_points)} points remain - reverting to original cloud.")
        return pcd

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    if new_colors is not None:
        new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    if SHOW_VISUALIZATIONS:
        print("Visualizing de-duplicated point cloud...")
        o3d.visualization.draw_geometries([new_pcd], window_name="De-duplicated point cloud")

    return new_pcd


def reconstruct_mesh_poisson(pcd: o3d.geometry.PointCloud, output_dir: str) -> o3d.geometry.TriangleMesh:
    print("\n" + "=" * 60)
    print("GENERATING MESH (Poisson)")
    print("=" * 60)

    if len(pcd.points) == 0:
        raise RuntimeError("Point cloud is empty, cannot reconstruct mesh.")

    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    print(f"Running Poisson surface reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
        scale=POISSON_SCALE,
        linear_fit=POISSON_LINEAR_FIT,
    )

    densities = np.asarray(densities)
    print("Filtering low-density vertices before cropping...")
    density_thresh = np.quantile(densities, DENSITY_FILTER_QUANTILE)
    vertices_to_remove = densities < density_thresh

    # IMPORTANT: Remove vertices on the original mesh, before cropping,
    # to avoid the vertex_mask size mismatch error.
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Crop using the point cloud's bounding box, slightly expanded
    print("Cropping mesh to point cloud bounding box...")
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())  # expand by 5 %
    mesh = mesh.crop(bbox)

    mesh.compute_vertex_normals()

    ensure_dir(os.path.join(output_dir, "mesh"))
    mesh_path = os.path.join(output_dir, "mesh", "mesh_poisson.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved mesh to: {mesh_path}")

    if SHOW_VISUALIZATIONS:
        print("Visualizing mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Mesh")

    return mesh



def generate_floor_plan(pcd: o3d.geometry.PointCloud, output_dir: str):

    print("\n" + "=" * 60)
    print("GENERATING FLOOR PLAN")
    print("=" * 60)

    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print("Point cloud empty, skipping floor plan.")
        return

    z_values = pts[:, 2]
    floor_z = np.percentile(z_values, 5)  # rough floor estimate

    slice_height = floor_z + FLOORPLAN_HEIGHT
    print(f"Slicing at {slice_height:.2f} m (floor ~ {floor_z:.2f} m)")

    mask = np.abs(pts[:, 2] - slice_height) <= SLICE_THICKNESS
    slice_points = pts[mask]

    print(f"Extracted {len(slice_points)} points in slice")

    # IMPORTANT: Guard against zero-size slice to avoid .min()/.max() errors.
    if slice_points.shape[0] == 0:
        print("No points in slice; skipping floor plan image generation.")
        return

    # Map slice points to 2D image coordinates
    xs = slice_points[:, 0]
    ys = slice_points[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Slight padding
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    width = height = FLOORPLAN_RESOLUTION
    img = np.zeros((height, width, 1), dtype=np.uint8)

    # Normalize to [0, 1], then to [0, width/height]
    x_norm = (xs - x_min) / (x_max - x_min + 1e-8)
    y_norm = (ys - y_min) / (y_max - y_min + 1e-8)

    u = (x_norm * (width - 1)).astype(np.int32)
    v = (1.0 - y_norm) * (height - 1)
    v = v.astype(np.int32)

    img[v, u, 0] = 255

    floorplan_dir = os.path.join(output_dir, "floorplan")
    ensure_dir(floorplan_dir)
    floorplan_path = os.path.join(floorplan_dir, "floorplan.png")
    cv2.imwrite(floorplan_path, img)
    print(f"Saved floor plan image: {floorplan_path}")

    if SHOW_VISUALIZATIONS:
        cv2.imshow("Floor Plan", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def main():
    output_dir = DEFAULT_OUTPUT_DIR

    print("=" * 60)
    print("STEP 2: RECONSTRUCTION FROM RAW POINT CLOUD")
    print("=" * 60)

    ensure_dir(output_dir)

    # 1. Load raw point cloud (exported by step1)
    pcd_raw = load_raw_pointcloud(RAW_POINTCLOUD_PATH)

    # 2. Filter and downsample
    pcd_filtered = downsample_and_filter_pointcloud(pcd_raw)

    # 3. Remove duplicate overlapping structures
    pcd_dedup = remove_duplicate_structures(pcd_filtered)

    # 4. Reconstruct mesh with Poisson and safe ordering
    mesh = reconstruct_mesh_poisson(pcd_dedup, output_dir)

    # 5. Generate floor plan from point cloud slice (safe against empty slice)
    generate_floor_plan(pcd_dedup, output_dir)

    print("\n✓ STEP 2 COMPLETE")
    print(f"Outputs written under: {output_dir}")


if __name__ == "__main__":
    main()
