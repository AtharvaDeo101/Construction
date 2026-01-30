import os
import json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

# Defaults
DEFAULT_SCAN_DIR = r"C:\Users\deoat\Desktop\Construct\data\scan_001"
DEFAULT_OUTPUT_DIR = r"C:\Users\deoat\Desktop\Construct\output"

# Point Cloud Parameters
DEPTH_TRUNC = 8.0       # Ignore depths beyond this
VOXEL_SIZE = 0.02       # Downsample resolution (2cm)
CONFIDENCE_THRESH = 0.5 # 0.0 to disable confidence filter

# Plane Detection Parameters
RANSAC_DISTANCE_THRESH = 0.05
RANSAC_N_POINTS = 3
RANSAC_ITERATIONS = 1000
MIN_PLANE_POINTS = 500

# Floor Plan Parameters
FLOORPLAN_HEIGHT = 1.2
WALL_THICKNESS = 0.15
SIMPLIFY_EPSILON = 0.05

SHOW_VISUALIZATIONS = True


def load_transforms(json_path: str):
    with open(json_path, 'r') as f:
        return json.load(f)


def create_output_dirs(output_dir: str):
    dirs = [
        "pointcloud",
        "mesh",
        "planes",
        "planes/plane_segments",
        "blueprint",
        "debug"
    ]
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)


def depth_to_pointcloud(depth_map, color_img, intrinsic_matrix, confidence_map=None):
    H, W = depth_map.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    depth = depth_map.flatten()

    if confidence_map is not None and CONFIDENCE_THRESH > 0.0:
        conf = confidence_map.flatten()
        valid_mask = (depth > 0) & (depth < DEPTH_TRUNC) & (conf > CONFIDENCE_THRESH)
    else:
        valid_mask = (depth > 0) & (depth < DEPTH_TRUNC)

    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    points = np.stack([X, Y, Z], axis=1)
    colors = color_img[v, u] / 255.0

    return points, colors


def transform_pointcloud(points, c2w_matrix):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_world = (c2w_matrix @ points_h.T).T
    return points_world[:, :3]


def fuse_pointclouds(scan_dir: str, output_dir: str):
    print("\n" + "="*60)
    print("PHASE A: POINT CLOUD FUSION")
    print("="*60)

    transforms_path = os.path.join(scan_dir, "transforms.json")
    transforms = load_transforms(transforms_path)

    frames = transforms['frames']
    print(f"Loading {len(frames)} frames...")

    all_points, all_colors = [], []

    for idx, frame in enumerate(frames):
        depth_path = os.path.join(scan_dir, frame['depth_path'])
        depth_map = np.load(depth_path)

        img_path = os.path.join(scan_dir, frame['file_path'])
        color_img = cv2.imread(img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        if color_img.shape[:2] != depth_map.shape:
            color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))

        confidence_map = None
        if frame.get('confidence_path'):
            conf_path = os.path.join(scan_dir, frame['confidence_path'])
            if os.path.exists(conf_path):
                confidence_map = np.load(conf_path)

        intrinsic = np.array(frame['intrinsic_matrix'])
        c2w = np.array(frame['transform_matrix'])

        points_cam, colors = depth_to_pointcloud(depth_map, color_img, intrinsic, confidence_map)
        points_world = transform_pointcloud(points_cam, c2w)

        all_points.append(points_world)
        all_colors.append(colors)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx+1}/{len(frames)} frames")

    print("\nMerging point clouds...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    print(f"Total points before filtering: {len(all_points):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    raw_path = os.path.join(output_dir, "pointcloud", "raw_cloud.ply")
    o3d.io.write_point_cloud(raw_path, pcd)
    print(f"✓ Saved raw point cloud: {raw_path}")

    print(f"\nDownsampling with voxel size {VOXEL_SIZE}m...")
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Points after downsampling: {len(pcd_down.points):,}")

    print("Removing statistical outliers...")
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    print(f"Points after filtering: {len(pcd_clean.points):,}")

    filtered_path = os.path.join(output_dir, "pointcloud", "filtered_cloud.ply")
    o3d.io.write_point_cloud(filtered_path, pcd_clean)
    print(f"✓ Saved filtered point cloud: {filtered_path}")

    print("Estimating normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)

    final_path = os.path.join(output_dir, "pointcloud", "colored_cloud.ply")
    o3d.io.write_point_cloud(final_path, pcd_clean)
    print(f"✓ Saved final point cloud with normals: {final_path}")

    if SHOW_VISUALIZATIONS:
        print("\nVisualizing point cloud...")
        o3d.visualization.draw_geometries([pcd_clean], window_name="Fused Point Cloud")

    return pcd_clean


def detect_planes(pcd, output_dir: str):
    print("\n" + "="*60)
    print("PHASE B: PLANE DETECTION")
    print("="*60)

    planes = []
    plane_clouds = []
    remaining_pcd = pcd

    max_planes = 10
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]

    for plane_idx in range(max_planes):
        if len(remaining_pcd.points) < MIN_PLANE_POINTS:
            break

        print(f"\nDetecting plane {plane_idx + 1}...")

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESH,
            ransac_n=RANSAC_N_POINTS,
            num_iterations=RANSAC_ITERATIONS
        )

        if len(inliers) < MIN_PLANE_POINTS:
            print(f"  Found only {len(inliers)} points, stopping.")
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        angle_with_vertical = np.arccos(np.abs(normal[2])) * 180 / np.pi

        if angle_with_vertical < 15:
            plane_type = "floor/ceiling"
        elif angle_with_vertical > 75:
            plane_type = "wall"
        else:
            plane_type = "sloped"

        print(f"  Plane {plane_idx + 1}: {len(inliers):,} points | Type: {plane_type}")
        print(f"  Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        print(f"  Distance (d): {d:.3f}m")

        plane_pcd = remaining_pcd.select_by_index(inliers)
        plane_pcd.paint_uniform_color(colors_palette[plane_idx % 10])

        planes.append({
            'id': plane_idx,
            'type': plane_type,
            'normal': normal.tolist(),
            'distance': float(d),
            'equation': [float(a), float(b), float(c), float(d)],
            'num_points': len(inliers)
        })
        plane_clouds.append(plane_pcd)

        plane_path = os.path.join(output_dir, "planes", "plane_segments", f"{plane_type}_{plane_idx}.ply")
        o3d.io.write_point_cloud(plane_path, plane_pcd)

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    print(f"\n✓ Detected {len(planes)} planes")

    planes_json_path = os.path.join(output_dir, "planes", "detected_planes.json")
    with open(planes_json_path, 'w') as f:
        json.dump({'planes': planes}, f, indent=2)
    print(f"✓ Saved plane data: {planes_json_path}")

    colored_planes = o3d.geometry.PointCloud()
    for plane_pcd in plane_clouds:
        colored_planes += plane_pcd

    viz_path = os.path.join(output_dir, "planes", "plane_visualization.ply")
    o3d.io.write_point_cloud(viz_path, colored_planes)
    print(f"✓ Saved plane visualization: {viz_path}")

    if SHOW_VISUALIZATIONS:
        print("\nVisualizing detected planes...")
        o3d.visualization.draw_geometries([colored_planes], window_name="Detected Planes")

    return planes, plane_clouds


def generate_mesh_bpa(pcd, output_dir: str):
    print("\n" + "="*60)
    print("GENERATING MESH (BALL PIVOTING)")
    print("="*60)

    print("Estimating average neighbor distance for BPA radii...")
    dists = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(dists)
    print(f"Average neighbor distance: {avg_dist:.4f} m")

    # Radii: small → detail, large → bridge small gaps
    radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
    print(f"Using BPA radii: {radii}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    mesh.compute_vertex_normals()

    mesh_path = os.path.join(output_dir, "mesh", "bpa_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"✓ Saved BPA mesh: {mesh_path}")

    if SHOW_VISUALIZATIONS:
        print("\nVisualizing BPA mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="BPA Mesh")

    return mesh


def generate_floor_plan(pcd, planes, output_dir: str):
    print("\n" + "="*60)
    print("PHASE C: BLUEPRINT GENERATION")
    print("="*60)

    points = np.asarray(pcd.points)

    floor_plane = None
    floor_z = None

    for plane in planes:
        normal = np.array(plane['normal'])
        if normal[2] > 0.8:
            a, b, c, d = plane['equation']
            dist = abs(d) / np.linalg.norm([a, b, c]) if c != 0 else abs(d)
            if floor_z is None or dist < floor_z:
                floor_z = dist
                floor_plane = plane

    if floor_plane is None:
        print("⚠️ Warning: Could not identify floor plane, using min Z")
        floor_z = points[:, 2].min()

    slice_height = floor_z + FLOORPLAN_HEIGHT
    print(f"Slicing at height: {slice_height:.2f}m (floor approx at {floor_z:.2f}m)")

    tolerance = 0.1
    mask = (points[:, 2] > slice_height - tolerance) & (points[:, 2] < slice_height + tolerance)
    slice_points = points[mask]

    if len(slice_points) < 100:
        print("⚠️ Few points at slice height, adjusting to median Z band...")
        slice_height = np.median(points[:, 2])
        mask = (points[:, 2] > slice_height - 0.2) & (points[:, 2] < slice_height + 0.2)
        slice_points = points[mask]

    print(f"Extracted {len(slice_points):,} points for floor plan")

    slice_2d = slice_points[:, :2]

    x_min, y_min = slice_2d.min(axis=0)
    x_max, y_max = slice_2d.max(axis=0)

    padding = 0.5
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    resolution = 100
    img_width = int((x_max - x_min) * resolution)
    img_height = int((y_max - y_min) * resolution)

    floor_plan_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    for point in slice_2d:
        px = int((point[0] - x_min) * resolution)
        py = int((y_max - point[1]) * resolution)
        if 0 <= px < img_width and 0 <= py < img_height:
            cv2.circle(floor_plan_img, (px, py), 2, (0, 0, 0), -1)

    kernel = np.ones((5, 5), np.uint8)
    floor_plan_gray = cv2.cvtColor(floor_plan_img, cv2.COLOR_BGR2GRAY)
    floor_plan_gray = cv2.morphologyEx(255 - floor_plan_gray, cv2.MORPH_CLOSE, kernel)
    floor_plan_img = cv2.cvtColor(255 - floor_plan_gray, cv2.COLOR_GRAY2BGR)

    plan_path = os.path.join(output_dir, "blueprint", "floorplan_2d.png")
    cv2.imwrite(plan_path, floor_plan_img)
    print(f"✓ Saved 2D floor plan: {plan_path}")

    blueprint_meta = {
        "resolution": resolution,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "img_width": img_width,
        "img_height": img_height,
    }
    meta_path = os.path.join(output_dir, "blueprint", "blueprint_meta.json")
    with open(meta_path, "w") as f:
        json.dump(blueprint_meta, f, indent=2)

    edges = cv2.Canny(floor_plan_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    wireframe_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    cv2.drawContours(wireframe_img, contours, -1, (0, 0, 0), 2)

    wireframe_path = os.path.join(output_dir, "blueprint", "wireframe_2d.png")
    cv2.imwrite(wireframe_path, wireframe_img)
    print(f"✓ Saved wireframe blueprint: {wireframe_path}")

    print(f"\n{'='*60}")
    print("BLUEPRINT GENERATION COMPLETE")
    print(f"{'='*60}\n")


def plot_camera_trajectory(scan_dir: str, output_dir: str):
    transforms_path = os.path.join(scan_dir, "transforms.json")
    transforms = load_transforms(transforms_path)

    positions = []
    for frame in transforms['frames']:
        c2w = np.array(frame['transform_matrix'])
        positions.append(c2w[:3, 3])

    positions = np.array(positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-o', markersize=4)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Trajectory')
    ax.legend()

    traj_path = os.path.join(output_dir, "debug", "camera_trajectory.png")
    plt.savefig(traj_path, dpi=150)
    print(f"✓ Saved camera trajectory: {traj_path}")

    if SHOW_VISUALIZATIONS:
        plt.show()
    plt.close(fig)


def run_step2(scan_dir: str,
              output_dir: str,
              show_visualizations: bool = True,
              use_bpa_mesh: bool = True) -> None:
    global SHOW_VISUALIZATIONS
    SHOW_VISUALIZATIONS = bool(show_visualizations)

    create_output_dirs(output_dir)
    print("\nGenerating debug visualizations...")
    plot_camera_trajectory(scan_dir, output_dir)
    pcd = fuse_pointclouds(scan_dir, output_dir)
    planes, _ = detect_planes(pcd, output_dir)
    if use_bpa_mesh:
        generate_mesh_bpa(pcd, output_dir)
    else:
        print("Mesh generation disabled.")
    generate_floor_plan(pcd, planes, output_dir)
    print("\n✓ STEP 2 COMPLETE!")
    print(f"Outputs saved to: {output_dir}")


def main():
    print("3D RECONSTRUCTION & BLUEPRINT GENERATION (BPA Mesh)")
    run_step2(DEFAULT_SCAN_DIR, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
